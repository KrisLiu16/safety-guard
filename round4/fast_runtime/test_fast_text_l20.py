"""L20-only audit of the fixed window classifier's graph text runtime.

Run after packaging:
  python test_fast_text_l20.py --bundle /work/output/fast_runtime/bundle
No training, data generation or candidate selection is performed. CPU checks
never import torch; an optional local tokenizer JSON validates the BPE fixture.
Normal execution is supervised in a separate process group with a time limit.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

CHECKPOINT_SHA256 = "bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2"
ROLES = ("user", "assistant")
LABELS = ("safe", "unsafe", "controversial")
LENGTHS = tuple(range(1, 33))
BLOCKS_PER_LENGTH = 3
APPENDS = 128
WHOLE_PROBABILITY_ATOL = .03
SAME_CACHE_PROBABILITY_ATOL = .001
STATE_TOLERANCES = {
    "torch.bfloat16": {"atol": .02, "rtol": .01},
    "torch.float32": {"atol": .002, "rtol": .001},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_module(name, path):
    path = Path(path).resolve()
    previous = sys.modules.get(name)
    if previous is not None:
        if Path(previous.__file__).resolve() != path:
            raise RuntimeError(f"Refusing cached module from another directory: {name}")
        return previous
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare_probabilities(expected, actual, tolerance):
    """Reject every invalid value before max(); NaN must never become a pass."""
    errors = {}
    for role in ROLES:
        try:
            a = [expected[role][label] for label in LABELS]
            b = [actual[role][label] for label in LABELS]
            valid = all(
                isinstance(v, (float, int)) and not isinstance(v, bool)
                and math.isfinite(v) and 0 <= v <= 1 for v in a + b
            ) and abs(sum(a) - 1) <= 1e-4 and abs(sum(b) - 1) <= 1e-4
            differences = [abs(x - y) for x, y in zip(a, b)] if valid else []
            errors[role] = max(differences) if differences and all(map(math.isfinite, differences)) else None
        except (KeyError, TypeError, ValueError, OverflowError):
            errors[role] = None
    # Preserve the prior graph test's <= .001 and text test's strict < .03.
    inclusive = tolerance == SAME_CACHE_PROBABILITY_ATOL
    return {"pass": all(v is not None and (v <= tolerance if inclusive else v < tolerance)
                        for v in errors.values()),
            "max_absolute_error_by_role": errors, "atol": tolerance,
            "comparison": "<=" if inclusive else "<"}


def summarize_performance(rows, initial_tokens, final_tokens, chunks, seconds):
    if len(rows) != APPENDS or len(chunks) != APPENDS or len(seconds) != APPENDS:
        raise ValueError("A measured trajectory must contain exactly 128 appends")
    if any(not math.isfinite(v) or v <= 0 for v in seconds):
        raise ValueError("Invalid measured duration")
    net = final_tokens - initial_tokens
    if net <= 0 or sum(row["net_new_tokens"] for row in rows) != net:
        raise ValueError("Net token accounting disagrees with accepted session state")
    elapsed = sum(seconds)
    latency = sorted(value * 1000 for value in seconds)
    runtime_seconds = [row["wall_seconds"] for row in rows]
    if any(not math.isfinite(v) or v < 0 for v in runtime_seconds):
        raise ValueError("Invalid runtime duration")
    return {
        "accepted_appends": len(rows), "initial_native_tokens": initial_tokens,
        "final_native_tokens": final_tokens, "net_new_native_tokens": net,
        "actual_forward_tokens": sum(row["forward_tokens"] for row in rows),
        "forward_calls": sum(row["forward_calls"] for row in rows),
        "replay_tokens": sum(row["replay_tokens"] for row in rows),
        "replayed_unchanged_prefix_tokens": sum(row["replayed_unchanged_prefix_tokens"] for row in rows),
        "rollback_events": sum(bool(row["rollback"]) for row in rows),
        "seconds": elapsed, "itps_net_native_input": net / elapsed,
        "classifications_per_second": len(rows) / elapsed,
        "input_utf8_bytes_per_second": sum(len(text.encode()) for text in chunks) / elapsed,
        "p50_ms": statistics.median(latency),
        "p95_ms": latency[math.ceil(.95 * len(latency)) - 1],
        "append_latency_ms": [value * 1000 for value in seconds],
        "runtime_reported_append_latency_ms": [value * 1000 for value in runtime_seconds],
        "runtime_reported_seconds": sum(runtime_seconds),
        "scope": "External wall clock around complete append_text; includes full retokenization, "
                 "cache/snapshot clones, rollback/replay, both CPU-visible risk heads and return "
                 "construction. Same loaded W512 weights, text trace and independent full-trace warmup. "
                 "Excludes loading, startup capture, session prefill, diagnostics, HTTP and input waiting. "
                 "Numerator is accepted net native tokens; replay and auxiliary startup tokens are separate.",
    }


def cpu_checks(tokenizer_json=None):
    directory = Path(__file__).resolve().parent
    sources = ("test_fast_text_l20.py", "graph_stream.py", "graph_text_runtime.py", "text_stream_runtime.py")
    for name in sources:
        ast.parse((directory / name).read_text())
    graph = load_module("graph_stream", directory / "graph_stream.py")
    load_module("text_stream_runtime", directory / "text_stream_runtime.py")
    runtime = load_module("graph_text_runtime", directory / "graph_text_runtime.py")
    for length in LENGTHS:
        graph.validate_parameters(chunk_tokens=length)
        graph.validate_ids(list(range(length)), 100, length)
        assert runtime.select_execution(512, length) == "graph"
        assert runtime.select_execution(8192 - length, length) == "graph"
    assert runtime.select_execution(511, 32) == "eager"
    assert runtime.select_execution(512, 33) == "eager"
    rejected = 0
    for length in (0, 33, True, -1, 1.5):
        try:
            graph.validate_parameters(chunk_tokens=length)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Invalid graph length accepted")
    for prefix, length in ((8192, 1), (8180, 13), (-1, 8), (512, 0)):
        try:
            runtime.select_execution(prefix, length)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Invalid scheduling boundary accepted")
    good = {role: dict(zip(LABELS, (.8, .1, .1))) for role in ROLES}
    assert compare_probabilities(good, good, .03)["pass"]
    for value in (float("nan"), float("inf"), -float("inf"), -1, 2, True):
        bad = copy.deepcopy(good)
        bad["assistant"]["unsafe"] = value
        check = compare_probabilities(good, bad, .03)
        assert not check["pass"] and check["max_absolute_error_by_role"]["assistant"] is None
        json.dumps(check, allow_nan=False)
        rejected += 1
    rows = [{"net_new_tokens": 2, "forward_tokens": 3, "forward_calls": 1, "replay_tokens": 1,
             "replayed_unchanged_prefix_tokens": 1, "rollback": True, "wall_seconds": .001} for _ in range(APPENDS)]
    stats = summarize_performance(rows, 600, 856, ["ab"] * APPENDS, [.002] * APPENDS)
    assert stats["net_new_native_tokens"] == 256 and stats["actual_forward_tokens"] == 384
    assert abs(stats["itps_net_native_input"] - 1000) < 1e-9
    assert stats["p95_ms"] == 2 and len(stats["append_latency_ms"]) == 128
    bpe = None
    if tokenizer_json:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        bpe = contraction_fixture(tokenizer)
        bpe = {key: value for key, value in bpe.items() if key != "text"}
    if "torch" in sys.modules:
        raise AssertionError("CPU-only audit imported torch")
    return {"status": "cpu_checks_only", "pass": True, "syntax_files": len(sources),
            "exact_lengths_checked": list(LENGTHS), "invalid_paths_rejected": rejected,
            "bpe_fixture": bpe, "model_calls": 0, "torch_imported": False,
            "neural_or_performance_pass_claimed": False}


def encode(tokenizer, text):
    # Match the runtime adapter, including rejecting configured truncation.
    module = sys.modules["text_stream_runtime"]
    module._reject_configured_truncation(tokenizer)
    encoded = tokenizer.encode(text, **module._encode_kwargs(tokenizer))
    if getattr(encoded, "overflowing", None):
        raise ValueError("Tokenizer truncated the fixture")
    return list(encoded.ids if hasattr(encoded, "ids") else encoded)


def contraction_fixture(tokenizer):
    text = "x " * 699 + "informatio"
    sequence = [encode(tokenizer, "USER:\n" + text + suffix) for suffix in ("", "n", "n!")]
    lengths = [len(ids) for ids in sequence]
    if lengths != [704, 703, 704]:
        raise AssertionError(f"Real-tokenizer post-window contraction fixture changed: {lengths}")
    if sequence[0] == sequence[1][:len(sequence[0])]:
        raise AssertionError("Fixture does not change an accepted BPE suffix")
    return {"text": text, "native_lengths": lengths,
            "token_ids_sha256": [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in sequence]}


class Audit:
    def __init__(self, torch, state_module, text_module, eager, fast, report, output):
        self.torch, self.state, self.text = torch, state_module, text_module
        self.eager, self.fast = eager, fast
        self.report, self.output = report, output

    def save(self):
        write_report(self.output, self.report)

    def add(self, name, passed, **details):
        self.report["checks"].append({"case": name, "pass": bool(passed), **details})
        self.save()

    def fingerprint(self, cache):
        if cache is None:
            return None
        digest = hashlib.sha256(json.dumps(self.state.state_metadata(cache), sort_keys=True).encode())
        for key, tensor in sorted(self.state.state_tensors(cache).items()):
            digest.update(str((key, tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.detach().contiguous().reshape(-1).view(self.torch.uint8).cpu().numpy().tobytes())
        return digest.hexdigest()

    def session_fingerprint(self, session):
        values = {"messages": session.messages, "ids": session.token_ids,
                  "probabilities": session._probabilities, "cache": self.fingerprint(session._cache),
                  "snapshots": [(snap.position, self.fingerprint(snap.cache), snap.probabilities)
                                for snap in session._snapshots],
                  "counters": [session._forward_tokens, session._replay_tokens, session._wall_seconds]}
        return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def compare_states(self, expected, actual):
        torch = self.torch
        left, right = self.state.state_tensors(expected), self.state.state_tensors(actual)
        topology = set(left) == set(right) and len(left) == 48
        metadata = self.state.state_metadata(expected) == self.state.state_metadata(actual)
        passed = topology and metadata
        families = {}
        if topology:
            for key in sorted(left):
                a, b = left[key], right[key]
                dtype = str(a.dtype)
                if a.shape != b.shape or a.dtype != b.dtype or dtype not in STATE_TOLERANCES:
                    return {"pass": False, "metadata_equal": metadata, "error": f"Unsupported state layout: {key}"}
                tolerance = STATE_TOLERANCES[dtype]
                finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                maximum = normalized = None
                if finite:
                    delta = (a.float() - b.float()).abs()
                    scaled = delta / (tolerance["atol"] + tolerance["rtol"] * a.float().abs())
                    finite = bool(torch.isfinite(delta).all() and torch.isfinite(scaled).all())
                    if finite:
                        maximum = float(delta.max())
                        normalized = float(scaled.max())
                valid = finite and normalized <= 1
                passed = passed and valid
                aggregate = families.setdefault(key[1], {"tensors": 0, "dtype": dtype, "finite": True,
                    "pass": True, "max_absolute_error": 0., "max_normalized_error": 0., "tolerance": tolerance})
                aggregate["tensors"] += 1
                aggregate["finite"] = aggregate["finite"] and finite
                aggregate["pass"] = aggregate["pass"] and valid
                if finite:
                    aggregate["max_absolute_error"] = max(aggregate["max_absolute_error"], maximum)
                    aggregate["max_normalized_error"] = max(aggregate["max_normalized_error"], normalized)
        return {"pass": bool(passed), "state_tensor_count": len(left), "topology_equal": topology,
                "metadata_equal": metadata, "families": families}

    def clone(self, cache):
        with self.torch.inference_mode():
            return copy.deepcopy(cache)

    def eager_forward(self, ids, cache=None):
        return self.eager._forward(ids, self.clone(cache))

    def arena_isolation(self, *sessions):
        inventory = self.text.audit_cache_tensor_isolation(*sessions)
        arena_pointers = set()
        for runner in self.fast._runners.values():
            pointers = self.state.storage_pointers(runner._arena)
            if arena_pointers & pointers:
                raise AssertionError("Graph arenas alias each other")
            arena_pointers.update(pointers)
        for session in sessions:
            for cache in [session._cache, *(snap.cache for snap in session._snapshots)]:
                if self.state.storage_pointers(cache) & arena_pointers:
                    raise AssertionError("Accepted session cache/snapshot aliases a graph arena")
        return {**inventory, "shared_graph_arenas_disjoint_from_sessions": True,
                "graph_arena_count": len(self.fast._runners)}

    def pair_check(self, name, a, b, *, whole=True):
        identities = (a.token_ids == b.token_ids and a.messages == b.messages
                      and a.snapshot_positions == b.snapshot_positions)
        direct = compare_probabilities(a._probabilities, b._probabilities, SAME_CACHE_PROBABILITY_ATOL)
        state = self.compare_states(a._cache, b._cache)
        snapshots = []
        for x, y in zip(a._snapshots, b._snapshots):
            snapshots.append({"position": x.position, "state": self.compare_states(x.cache, y.cache),
                              "probabilities": compare_probabilities(x.probabilities, y.probabilities,
                                                                    SAME_CACHE_PROBABILITY_ATOL)})
        detail = {"identities_equal": identities, "eager_vs_graph_probabilities": direct,
                  "eager_vs_graph_state": state, "snapshots": snapshots}
        passed = identities and direct["pass"] and state["pass"] and all(
            item["state"]["pass"] and item["probabilities"]["pass"] for item in snapshots)
        if whole:
            whole_cache, whole_probs = self.eager_forward(b.token_ids)
            whole_probability = compare_probabilities(whole_probs, b._probabilities, WHOLE_PROBABILITY_ATOL)
            whole_state = self.compare_states(whole_cache, b._cache)
            detail.update(whole_probabilities=whole_probability, whole_state=whole_state)
            passed = passed and whole_probability["pass"] and whole_state["pass"]
        detail["isolation"] = self.arena_isolation(a, b)
        self.add(name, passed, **detail)


def run_exact_lengths(audit, tokenizer):
    audit.report["phase"] = "all_32_exact_lengths"
    audit.save()
    sequence = encode(tokenizer, "USER:\n" + "".join(
        f"第{i}条公开资料，核对目录和日期。 Public reading record {i}: organize ordinary notes.\n"
        for i in range(700)))
    if len(sequence) < 8192:
        raise AssertionError("Insufficient deterministic native-token test data")
    audit.report["native_trace_sha256"] = hashlib.sha256(json.dumps(sequence[:8192]).encode()).hexdigest()
    seed, _ = audit.eager_forward(sequence[:512])
    for length in LENGTHS:
        g_cache, e_cache = audit.clone(seed), audit.clone(seed)
        prefix = sequence[:512]
        blocks_seen = set()
        for block in range(BLOCKS_PER_LENGTH):
            # Distinct block locations and content for each length and iteration.
            start = 1024 + length * 97 + block * 41
            ids = sequence[start:start + length]
            while tuple(ids) in blocks_seen:
                start += 1
                ids = sequence[start:start + length]
            if len(ids) != length:
                raise AssertionError("Not enough distinct blocks for this graph length")
            blocks_seen.add(tuple(ids))
            prefix = [*prefix, *ids]
            old_digest = audit.fingerprint(g_cache)
            same_cache, same_probs = audit.eager_forward(ids, g_cache)
            before = audit.fast.stats()
            next_cache, graph_probs = audit.fast._forward(ids, g_cache)
            after = audit.fast.stats()
            untouched = old_digest == audit.fingerprint(g_cache)
            route_correct = (after["graph_calls"] - before["graph_calls"] == 1
                             and after["graph_forward_tokens"] - before["graph_forward_tokens"] == length
                             and after["eager_calls"] == before["eager_calls"])
            e_cache, eager_probs = audit.eager_forward(ids, e_cache)
            whole_cache, whole_probs = audit.eager_forward(prefix)
            comparisons = {
                "same_input_cache_probabilities": compare_probabilities(same_probs, graph_probs, SAME_CACHE_PROBABILITY_ATOL),
                "same_input_cache_state": audit.compare_states(same_cache, next_cache),
                "independent_eager_trajectory_probabilities": compare_probabilities(eager_probs, graph_probs, SAME_CACHE_PROBABILITY_ATOL),
                "independent_eager_trajectory_state": audit.compare_states(e_cache, next_cache),
                "whole_probabilities": compare_probabilities(whole_probs, graph_probs, WHOLE_PROBABILITY_ATOL),
                "whole_state": audit.compare_states(whole_cache, next_cache),
            }
            pointers = audit.state.storage_pointers(next_cache)
            no_alias = not pointers.intersection(audit.state.storage_pointers(g_cache))
            no_alias = no_alias and all(not pointers.intersection(audit.state.storage_pointers(r._arena))
                                       for r in audit.fast._runners.values())
            audit.add(f"exact_length_{length}_block_{block}", untouched and route_correct and no_alias
                      and all(value["pass"] for value in comparisons.values()), length=length,
                      block=block, prefix_tokens=len(prefix), input_cache_bitwise_unchanged=untouched,
                      graph_route_confirmed=route_correct, output_storage_isolated=no_alias, **comparisons)
            g_cache = next_cache
        del g_cache, e_cache, whole_cache, same_cache, next_cache
    # Same saved state can be restored via copy-in, regardless of the last graph
    # length used. Replaying it twice must not alias or mutate the original.
    saved = audit.clone(seed)
    original = audit.fingerprint(saved)
    replay_a, p_a = audit.fast._forward(sequence[512:529], saved)
    audit.fast._forward(sequence[2000:2032], audit.clone(seed))
    replay_b, p_b = audit.fast._forward(sequence[512:529], saved)
    comparison = audit.compare_states(replay_a, replay_b)
    probabilities = compare_probabilities(p_a, p_b, SAME_CACHE_PROBABILITY_ATOL)
    audit.add("raw_snapshot_restore_replay_after_another_length",
              original == audit.fingerprint(saved) and comparison["pass"] and probabilities["pass"],
              snapshot_bitwise_unchanged=original == audit.fingerprint(saved),
              state=comparison, probabilities=probabilities)
    # Near the maximum logical position: one 32-token block and mixed lengths
    # independently land at exactly 8192, exercising dynamic position buffers.
    near_seed, _ = audit.eager_forward(sequence[:8160])
    for schedule in ([32], [1, 2, 3, 4, 5, 6, 11]):
        g_cache, e_cache, position = audit.clone(near_seed), audit.clone(near_seed), 8160
        for length in schedule:
            ids = sequence[position:position + length]
            e_cache, e_probs = audit.eager_forward(ids, e_cache)
            g_cache, g_probs = audit.fast._forward(ids, g_cache)
            position += length
            same = audit.compare_states(e_cache, g_cache)
            probability = compare_probabilities(e_probs, g_probs, SAME_CACHE_PROBABILITY_ATOL)
            audit.add(f"near_8192_{schedule}_{position}", same["pass"] and probability["pass"],
                      logical_tokens=position, schedule=schedule, state=same, probabilities=probability)
        whole, probs = audit.eager_forward(sequence[:8192])
        state = audit.compare_states(whole, g_cache)
        probability = compare_probabilities(probs, g_probs, WHOLE_PROBABILITY_ATOL)
        audit.add(f"whole_8192_{schedule}", state["pass"] and probability["pass"],
                  state=state, probabilities=probability)
        original = audit.fingerprint(g_cache)
        stats = audit.fast.stats()
        try:
            audit.fast._forward(sequence[:1], g_cache)
        except ValueError:
            rejected = True
        else:
            rejected = False
        audit.add(f"raw_8193_rejected_{schedule}",
                  rejected and original == audit.fingerprint(g_cache) and audit.fast.stats() == stats,
                  rejection_before_graph_execution=rejected, engine_remains_ready=audit.fast.stats()["phase"] == "ready")


def run_text_correctness(audit, tokenizer):
    audit.report["phase"] = "real_text_transactions"
    audit.save()
    # The graph subclass must retain the eager path before a full window exists,
    # including an empty message header and single-character BPE updates.
    early_a, early_b = audit.eager.new_session(), audit.fast.new_session()
    before = audit.fast.stats()
    early_a.begin_message("user", "")
    early_b.begin_message("user", "")
    for character in "中立。information":
        early_a.append_text(character)
        early_b.append_text(character)
    audit.pair_check("early_prefix_eager_character_input", early_a, early_b)
    after = audit.fast.stats()
    audit.add("early_prefix_never_uses_graph", after["graph_calls"] == before["graph_calls"]
              and after["eager_calls"] > before["eager_calls"])
    blank = audit.fast.new_session()
    before = audit.fast.stats()
    try:
        blank.append_text("no message yet")
    except ValueError:
        rejected = True
    else:
        rejected = False
    audit.add("append_before_message_rejected", rejected and blank.token_ids == []
              and blank.messages == [] and audit.fast.stats() == before)
    fixture = contraction_fixture(tokenizer)
    audit.report["post_window_bpe_fixture"] = {key: value for key, value in fixture.items() if key != "text"}
    a, b = audit.eager.new_session(), audit.fast.new_session()
    first_a = a.begin_message("user", fixture["text"])
    first_b = b.begin_message("user", fixture["text"])
    audit.pair_check("bpe_704_before", a, b)
    before_other = audit.session_fingerprint(b)
    other_a, other_b = audit.eager.new_session(), audit.fast.new_session()
    background = "另一会话的一般阅读内容。 Separate reading session with dates and notes.\n" * 60
    other_a.begin_message("assistant", background)
    other_b.begin_message("assistant", background)
    audit.add("new_session_does_not_mutate_existing", before_other == audit.session_fingerprint(b))
    for suffix, expected in (("n", 703), ("!", 704)):
        frozen_other = audit.session_fingerprint(other_b)
        r_a, r_b = a.append_text(suffix), b.append_text(suffix)
        audit.pair_check(f"bpe_to_{expected}", a, b)
        counter_keys = ("rollback", "restore_position", "fallback_full_replay",
                        "forward_tokens", "replay_tokens", "snapshot_positions")
        expected_snapshots = [576, 640] if expected == 703 else [640, 704]
        rollback_correct = (expected != 703 or (r_b["rollback"] and r_b["fallback_full_replay"]
                                              and r_b["restore_position"] == 0))
        audit.add(f"bpe_lengths_and_other_session_{expected}",
                  r_a["native_tokens"] == expected == r_b["native_tokens"]
                  and frozen_other == audit.session_fingerprint(other_b) and rollback_correct
                  and r_b["snapshot_positions"] == expected_snapshots
                  and all(r_a[key] == r_b[key] for key in counter_keys),
                  eager_counters={key: r_a[key] for key in counter_keys},
                  graph_counters={key: r_b[key] for key in counter_keys},
                  expected_snapshot_positions=expected_snapshots)
    if not (first_a["native_tokens"] == first_b["native_tokens"] == 704):
        raise AssertionError("Initial BPE fixture length changed")
    for index, chunk in enumerate(("补充说明。", " informatio", "n", "!", " Ordinary information.", "\n下一段")):
        frozen = audit.session_fingerprint(other_b)
        a.append_text(chunk)
        b.append_text(chunk)
        audit.add(f"interleave_other_unchanged_{index}", frozen == audit.session_fingerprint(other_b))
        frozen = audit.session_fingerprint(b)
        other_a.append_text(f" 条目 {index}.")
        other_b.append_text(f" 条目 {index}.")
        audit.add(f"interleave_primary_unchanged_{index}", frozen == audit.session_fingerprint(b))
        audit.pair_check(f"interleave_primary_{index}", a, b, whole=index == 5)
        audit.pair_check(f"interleave_other_{index}", other_a, other_b, whole=index == 5)
    audit.add("two_sessions_and_all_snapshots_isolated", True,
              inventory=audit.arena_isolation(a, b, other_a, other_b))
    for role, text in (("assistant", "按上下文说明中立事实。"), ("user", "What does the word mean?")):
        a.begin_message(role, text)
        b.begin_message(role, text)
        audit.pair_check(f"role_switch_{role}", a, b)
    before = audit.fast.stats()
    empty = b.append_text("")
    audit.add("empty_append_no_forward",
              empty["forward_tokens"] == 0 and b.token_ids == a.token_ids
              and audit.fast.stats() == before and b._forward_tokens == a._forward_tokens,
              forward_tokens=empty["forward_tokens"])
    # Empty appends update accepted timing counters, so begin a fresh transaction
    # fingerprint after the successful no-op.
    def reject_unchanged(name, call, exception, required_text=None):
        old = audit.session_fingerprint(b)
        try:
            call()
        except exception as error:
            rejected = required_text is None or required_text in str(error)
        else:
            rejected = False
        audit.add(name, rejected and old == audit.session_fingerprint(b),
                  rejected=rejected, complete_committed_state_unchanged=old == audit.session_fingerprint(b))

    reject_unchanged("overlength_transaction", lambda: b.append_text("超长文本。" * 10000), ValueError)
    reject_unchanged("invalid_role_transaction", lambda: b.begin_message("system", "text"), ValueError)
    reject_unchanged("invalid_type_transaction", lambda: b.append_text(None), ValueError)
    normal = b._runner
    for mode in ("raise_after_forward", "nonfinite_after_forward"):
        invoked = []
        def broken(ids, cache, mode=mode):
            updated, probabilities = normal(ids, cache)
            invoked.append(True)
            if mode == "raise_after_forward":
                raise RuntimeError("intentional failure after real forward")
            probabilities["assistant"]["unsafe"] = float("nan")
            return updated, probabilities
        b._runner = broken
        try:
            reject_unchanged(mode, lambda: b.append_text(" 中立。"),
                             RuntimeError if mode == "raise_after_forward" else ValueError)
        finally:
            b._runner = normal
        audit.add(mode + "_actually_forwarded", bool(invoked))
        a.append_text(" 恢复正常。")
        b.append_text(" 恢复正常。")
        audit.pair_check(mode + "_retry", a, b)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "enable_truncation"):
        raise RuntimeError("Real tokenizer backend is required for truncation rejection audit")
    backend.enable_truncation(16)
    try:
        reject_unchanged("configured_tokenizer_truncation", lambda: b.append_text("更多文本"), ValueError)
    finally:
        backend.no_truncation()
    external_messages, external_ids = b.messages, b.token_ids
    old = audit.session_fingerprint(b)
    external_messages[-1]["content"] = "mutated external copy"
    external_ids[0] = -1
    audit.add("returned_messages_and_ids_do_not_alias", old == audit.session_fingerprint(b))
    audit.pair_check("all_transactions_final", a, b)
    # The API result's probability mapping is also a copy.
    returned = b.append_text("")
    original_probs = copy.deepcopy(b._probabilities)
    returned["risk_probabilities"]["unsafe"] = float("nan")
    audit.add("returned_probabilities_do_not_alias", b._probabilities == original_probs)


def run_text_performance(audit):
    audit.report["phase"] = "same_text_128_append_performance"
    audit.save()
    base = "".join(f"公开材料第{i}段讨论日期与目录。 Public note {i}: organize ordinary reading.\n" for i in range(40))
    variants = (" 中立资料。", " Public", " information.", " 说明含义。", "\nNew record:", " a", " 目录与日期。", " ordinary text.")
    chunks = [variants[index % len(variants)] for index in range(APPENDS)]
    trace = {"initial_messages": [{"role": "user", "content": base}], "appends": chunks}
    audit.report["performance_trace_sha256"] = hashlib.sha256(
        json.dumps(trace, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    audit.report["performance_trace"] = trace
    results, final_sessions = {}, {}
    for name, runtime in (("eager", audit.eager), ("graph", audit.fast)):
        warm = runtime.new_session()
        warm.begin_message("user", base)
        for text in chunks:
            warm.append_text(text)
        del warm
        session = runtime.new_session()
        session.begin_message("user", base)
        initial = len(session.token_ids)
        if not 512 <= initial < 8192:
            raise AssertionError("Performance prefix must have a full window")
        session_before = audit.text.audit_cache_tensor_isolation(session)
        counters_before = audit.fast.stats() if name == "graph" else None
        audit.torch.cuda.synchronize()
        rows, durations = [], []
        for text in chunks:
            started = time.perf_counter()
            row = session.append_text(text)
            durations.append(time.perf_counter() - started)
            # Consumption and finite checks are identical; the CPU conversion
            # itself already happened inside the timed append call.
            if not compare_probabilities(session._probabilities, session._probabilities, .001)["pass"]:
                raise RuntimeError("Invalid CPU-visible output in measured trajectory")
            rows.append(row)
        result = summarize_performance(rows, initial, len(session.token_ids), chunks, durations)
        result["per_append_accounting"] = [{key: row[key] for key in (
            "native_tokens", "net_new_tokens", "forward_tokens", "forward_calls", "replay_tokens",
            "replayed_unchanged_prefix_tokens", "rollback", "restore_position", "snapshot_positions")} for row in rows]
        result["session_storage_before"] = session_before
        result["session_storage_after"] = audit.text.audit_cache_tensor_isolation(session)
        result["final_token_ids_sha256"] = hashlib.sha256(json.dumps(session.token_ids).encode()).hexdigest()
        result["final_probabilities_by_role"] = copy.deepcopy(session._probabilities)
        if counters_before is not None:
            result["request_counter_delta"] = {key: audit.fast.stats()[key] - value
                for key, value in counters_before.items() if isinstance(value, int)}
            delta = result["request_counter_delta"]
            if (delta["graph_forward_tokens"] + delta["eager_forward_tokens"] != result["actual_forward_tokens"]
                    or delta["graph_calls"] + delta["eager_calls"] != result["forward_calls"]
                    or delta["graph_calls"] == 0):
                raise AssertionError("Measured graph/eager dispatch counts disagree with session accounting")
        results[name], final_sessions[name] = result, session
        audit.report["text_performance"] = results
        audit.save()
    audit.pair_check("measured_trajectory_final", final_sessions["eager"], final_sessions["graph"])
    left, right = results["eager"], results["graph"]
    accounting = all(left[key] == right[key] for key in (
        "net_new_native_tokens", "actual_forward_tokens", "forward_calls", "replay_tokens",
        "rollback_events", "final_token_ids_sha256", "per_append_accounting"))
    bounded = all(item["session_storage_before"]["total_tensor_storage_bytes"]
                  == item["session_storage_after"]["total_tensor_storage_bytes"] for item in results.values())
    audit.add("same_text_performance_accounting", accounting and bounded,
              native_and_replay_accounting_equal=accounting, per_session_state_bounded=bounded)
    audit.report["performance_comparison"] = {
        "graph_over_eager_net_itps": right["itps_net_native_input"] / left["itps_net_native_input"],
        "eager_over_graph_p95": left["p95_ms"] / right["p95_ms"],
        "same_model_same_device_same_trace": True,
        "order": ["eager", "graph"], "independent_complete_warmup_appends_each": APPENDS,
        "speedup_is_not_a_correctness_gate": True,
        "limitation": "One sequential 128-append measurement per engine; not a randomized concurrency or service-load benchmark.",
    }


def run_worker(args):
    report = {"status": "running", "pass": False, "phase": "verify_bundle_and_hardware",
              "variant": "window", "expected_checkpoint_sha256": CHECKPOINT_SHA256,
              "audit_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
              "quality_candidate_changed": False, "training_or_weights_modified": False,
              "generated_tokens": 0, "checks": [], "required_exact_lengths": list(LENGTHS),
              "blocks_per_length": BLOCKS_PER_LENGTH, "measured_appends_per_engine": APPENDS,
              "tolerances": {"whole_probability_atol_exclusive": WHOLE_PROBABILITY_ATOL,
                             "same_cache_probability_atol_inclusive": SAME_CACHE_PROBABILITY_ATOL,
                             "state": STATE_TOLERANCES}}
    write_report(args.output, report)
    fast = None
    began = time.perf_counter()
    try:
        bundle = args.bundle.resolve()
        loader = load_module("standalone_model", bundle / "standalone_model.py")
        # load_bundle verifies the entire frozen file manifest before model load,
        # asserts exactly one L20 and strictly loads the full classifier state.
        started = time.perf_counter()
        torch, model, tokenizer, metadata = loader.load_bundle(bundle)
        report["model_load_seconds"] = time.perf_counter() - started
        report["model_metadata"] = metadata
        if metadata.get("variant") != "window" or metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            raise RuntimeError("This audit only accepts the fixed window/SFT checkpoint")
        report["checkpoint_sha256_independent"] = sha256(bundle / "classifier.safetensors")
        if report["checkpoint_sha256_independent"] != CHECKPOINT_SHA256:
            raise RuntimeError("Actual checkpoint bytes do not match the fixed SHA")
        if torch.cuda.device_count() != 1 or "L20" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Exactly one visible L20 is required")
        torch.set_num_threads(4)
        text = load_module("text_stream_runtime", bundle / "text_stream_runtime.py")
        state = load_module("graph_stream", bundle / "graph_stream.py")
        graph_text = load_module("graph_text_runtime", bundle / "graph_text_runtime.py")
        report["loaded_sources"] = {name: {"path": str(Path(module.__file__).resolve()),
                                          "sha256": sha256(module.__file__)}
            for name, module in (("standalone_model", loader), ("text_stream_runtime", text),
                                 ("graph_stream", state), ("graph_text_runtime", graph_text),
                                 ("window_attention", sys.modules["window_attention"]))}
        eager = text.TextStreamRuntime(model, tokenizer)
        report["phase"] = "capture_all_lengths_at_startup"
        write_report(args.output, report)
        torch.cuda.reset_peak_memory_stats()
        startup_allocated = torch.cuda.memory_allocated()
        startup_reserved = torch.cuda.memory_reserved()
        started = time.perf_counter()
        fast = graph_text.GraphTextStreamRuntime(model, tokenizer)
        torch.cuda.synchronize()
        report["startup"] = {**fast.engine_metadata, "external_wall_seconds": time.perf_counter() - started,
            "allocated_before_bytes": startup_allocated, "reserved_before_bytes": startup_reserved,
            "allocated_after_bytes": torch.cuda.memory_allocated(), "reserved_after_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "allocation_scope": "Model already loaded; graph initialization only. Shared arenas/pools are not session caches."}
        audit = Audit(torch, state, text, eager, fast, report, args.output)
        metadata = fast.engine_metadata
        arena_tensors = [tensor for runner in fast._runners.values()
                         for tensor in state.state_tensors(runner._arena).values()]
        io_tensors = [tensor for runner in fast._runners.values()
                      for tensor in (runner._ids, runner._positions, runner._offsets, runner._output)]
        arena_bytes = state.unique_storage_bytes(arena_tensors)
        report["startup"]["shared_arena_storage_bytes_independent"] = arena_bytes
        report["startup"]["shared_arena_and_io_storage_bytes_independent"] = state.unique_storage_bytes(
            [*arena_tensors, *io_tensors])
        del arena_tensors, io_tensors
        initialized = (sorted(fast._runners) == list(LENGTHS)
                       and metadata["captured_chunk_lengths"] == list(LENGTHS)
                       and metadata["startup_forward_calls"] == 129
                       and metadata["auxiliary_input_tokens"] == 2624
                       and metadata["shared_graph_arena_storage_bytes"] == arena_bytes
                       and fast.stats()["graph_calls"] == fast.stats()["eager_calls"] == 0
                       and fast.stats()["phase"] == "ready")
        seeds = [runner.snapshot() for runner in fast._runners.values()]
        seed_fingerprints = [audit.fingerprint(seed) for seed in seeds]
        arena_pointers = set().union(*(state.storage_pointers(runner._arena) for runner in fast._runners.values()))
        snapshot_pointers = set()
        snapshots_disjoint = True
        for seed in seeds:
            pointers = state.storage_pointers(seed)
            snapshots_disjoint = snapshots_disjoint and not pointers.intersection(arena_pointers | snapshot_pointers)
            snapshot_pointers.update(pointers)
        seed_ids, _ = graph_text.seed_token_ids(tokenizer)
        reference_seed, _ = audit.eager_forward(seed_ids)
        restored = [audit.compare_states(reference_seed, cache) for cache in seeds]
        audit.add("startup_all_32_arenas_restored_to_independent_seed",
                  initialized and snapshots_disjoint and len(set(seed_fingerprints)) == 1
                  and all(row["pass"] for row in restored),
                  startup_accounting_correct=initialized, all_arena_seed_values_bitwise_equal=len(set(seed_fingerprints)) == 1,
                  snapshots_disjoint_from_arenas_and_each_other=snapshots_disjoint,
                  restored_seed_state=restored, seed_logical_tokens=[cache.get_seq_length() for cache in seeds])
        del seeds, reference_seed
        run_exact_lengths(audit, tokenizer)
        run_text_correctness(audit, tokenizer)
        run_text_performance(audit)
        audit.add("no_hidden_request_capture", fast.engine_metadata == metadata
                  and all(runner.storage_metrics()["setup_forward_calls"] == 4 for runner in fast._runners.values()),
                  final_request_stats=fast.stats())
        exact = [row for row in report["checks"] if row["case"].startswith("exact_length_")]
        coverage = len(exact) == len(LENGTHS) * BLOCKS_PER_LENGTH and all(
            sum(row["length"] == length for row in exact) == BLOCKS_PER_LENGTH for length in LENGTHS)
        audit.add("required_length_coverage", coverage, exact_case_count=len(exact))
        report.update(status="completed", phase="finished", pass_=all(row["pass"] for row in report["checks"]))
        report["pass"] = report.pop("pass_")
    except Exception as error:
        report.update(status="failed", pass_=False, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        report["pass"] = report.pop("pass_")
        traceback.print_exc()
    finally:
        if fast is not None:
            try:
                fast.close()
                report["graph_cleanup"] = "completed"
            except Exception as error:
                report.update(pass_=False, graph_cleanup="failed", cleanup_error=str(error))
                report["pass"] = report.pop("pass_")
        report["elapsed_seconds"] = time.perf_counter() - began
        write_report(args.output, report)
    return 0 if report["pass"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("/work/output/fast_runtime/bundle"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/fast_runtime/fast_text_audit.json"))
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--cpu-check-only", action="store_true")
    parser.add_argument("--tokenizer-json", type=Path, help="Optional CPU-only real native tokenizer fixture check")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 3600:
        parser.error("--timeout-seconds must be in 1..3600")
    if args.cpu_check_only:
        print(json.dumps(cpu_checks(args.tokenizer_json), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if args.worker:
        return run_worker(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Do not leave an earlier pass visible if this invocation fails to start.
    write_report(args.output, {"status": "starting", "pass": False,
                              "expected_checkpoint_sha256": CHECKPOINT_SHA256, "generated_tokens": 0})
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--bundle", str(args.bundle),
               "--output", str(args.output), "--timeout-seconds", str(args.timeout_seconds)]
    log = args.output.with_suffix(".log")
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # A descendant may outlive the leader; always signal the whole group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            report = json.loads(args.output.read_text())
            report.update(status="timed_out", pass_=False, timeout_seconds=args.timeout_seconds,
                          log=str(log), generated_tokens=0)
            report["pass"] = report.pop("pass_")
            write_report(args.output, report)
            return 1
    report = json.loads(args.output.read_text())
    if returncode != 0 or report.get("status") != "completed" or report.get("pass") is not True:
        report.update(pass_=False, worker_returncode=returncode, log=str(log))
        report["pass"] = report.pop("pass_")
        if report.get("status") in ("starting", "running"):
            report["status"] = "worker_failed"
        write_report(args.output, report)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
