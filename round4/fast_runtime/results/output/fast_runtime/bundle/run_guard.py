"""JSONL whole-message and actual text-append inference for a Round4 bundle.

Each input line is one operation: whole, begin_message, append_text, or
end_session. Output lines contain direct risk probabilities, never generated
tokens. All neural work requires one L20; --verify-only is CPU-only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

from standalone_model import INFERENCE_ENGINE, RISK_LABELS, ROLES, load_bundle, valid_threshold, verify_bundle


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--input", type=Path, default=Path("-"), help="UTF-8 JSONL, default stdin")
    parser.add_argument("--output", type=Path, default=Path("-"), help="UTF-8 JSONL, default stdout")
    parser.add_argument("--chunk-tokens", type=int, default=32)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=2048)
    parser.add_argument("--max-sessions", type=int, default=4)
    parser.add_argument("--inference-engine", choices=(INFERENCE_ENGINE, "eager"), default=INFERENCE_ENGINE,
                        help="Default: pre-captured window graphs for short filled-window text appends; eager is an explicit control")
    parser.add_argument("--threshold", type=float, help="Explicit uncalibrated binary threshold override")
    parser.add_argument("--verify-only", action="store_true", help="Check files/hashes; no torch/model call")
    args = parser.parse_args(argv)
    if (args.chunk_tokens < 1 or args.prefill_chunk_tokens < 1
            or not 1 <= args.max_sessions <= 64):
        parser.error("Chunk sizes must be positive; --max-sessions must be 1..64")
    if args.threshold is not None and not valid_threshold(args.threshold):
        parser.error("--threshold must be finite and in [0, 1]")
    if str(args.input) != "-" and args.input.resolve() == args.output.resolve():
        parser.error("Input and output must not refer to the same file")
    return args


def validate_messages(value):
    if not isinstance(value, list) or not value:
        raise ValueError("whole requires a nonempty messages array")
    result = []
    for message in value:
        if (not isinstance(message, dict) or message.get("role") not in ("system", *ROLES)
                or not isinstance(message.get("content"), str)):
            raise ValueError("Each message requires a system/user/assistant role and string content")
        result.append({"role": message["role"], "content": message["content"]})
    if result[-1]["role"] not in ROLES:
        raise ValueError("The final message must be user or assistant")
    return result


def threshold_result(selection, language, role, probabilities, explicit, streaming):
    if explicit is not None:
        value, kind = explicit, "explicit_uncalibrated"
    elif language is None:
        value, kind = None, "language_not_provided"
    else:
        value = selection["thresholds"].get(f"{language}/{role}")
        kind = "selected_checkpoint_calibration" if value is not None else "stratum_not_calibrated"
        if streaming and value is not None:
            kind = "whole_input_calibration_applied_to_prefix_unvalidated"
    return {"target_binary_decision": None if value is None else (
                "unsafe" if probabilities["unsafe"] >= value else "safe"),
            "threshold": {"kind": kind, "value": value,
                          "stratum": None if language is None else f"{language}/{role}"},
            "safe_prefix_release_validated": False}


class JSONLGuard:
    def __init__(self, args, torch, model, tokenizer, metadata):
        # Explicit path import ensures the text runtime is the checksum-verified
        # bundled code even when called from an experiment's Python process.
        from standalone_model import _load_module
        bundle = Path(metadata["bundle"])
        module = _load_module("text_stream_runtime", bundle / "text_stream_runtime.py")
        self.inference_engine = getattr(args, "inference_engine", INFERENCE_ENGINE)
        if metadata.get("inference_engine") != INFERENCE_ENGINE:
            raise ValueError("This entry point requires a window_cuda_graph bundle")
        if self.inference_engine == INFERENCE_ENGINE:
            _load_module("graph_stream", bundle / "graph_stream.py")
            graph_module = _load_module("graph_text_runtime", bundle / "graph_text_runtime.py")
            self.runtime = graph_module.GraphTextStreamRuntime(model, tokenizer, chunk_tokens=args.chunk_tokens,
                                                               prefill_chunk_tokens=args.prefill_chunk_tokens)
            self.engine_metadata = self.runtime.engine_metadata
        elif self.inference_engine == "eager":
            self.runtime = module.TextStreamRuntime(model, tokenizer, chunk_tokens=args.chunk_tokens,
                                                   prefill_chunk_tokens=args.prefill_chunk_tokens)
            self.engine_metadata = {"inference_engine": "eager", "capture_seconds": 0.,
                                    "auxiliary_input_tokens": 0, "shared_graph_arena_storage_bytes": 0,
                                    "captured_chunk_lengths": [],
                                    "initialization_excluded_from_request_accounting": True}
        else:
            raise ValueError("Unknown inference engine")
        self.serialize = module.serialize
        self.args, self.torch, self.model, self.tokenizer = args, torch, model, tokenizer
        self.metadata = metadata
        self.sessions = {}

    def _engine_stats(self):
        return self.runtime.stats() if self.inference_engine == INFERENCE_ENGINE else None

    def _request_execution(self, before, result):
        if before is None:
            delta = {"graph_calls": 0, "graph_forward_tokens": 0,
                     "eager_calls": result["forward_calls"], "eager_forward_tokens": result["forward_tokens"]}
        else:
            after = self._engine_stats()
            delta = {key: after[key] - before[key] for key in
                     ("graph_calls", "graph_forward_tokens", "eager_calls", "eager_forward_tokens")}
        if (any(not isinstance(value, int) or value < 0 for value in delta.values())
                or delta["graph_calls"] + delta["eager_calls"] != result["forward_calls"]
                or delta["graph_forward_tokens"] + delta["eager_forward_tokens"] != result["forward_tokens"]):
            raise RuntimeError("Engine call counters do not match the successful text transaction")
        return {"inference_engine": self.inference_engine, **delta,
                "initialization_included": False}

    def close(self):
        self.sessions.clear()
        close = getattr(self.runtime, "close", None)
        if close is not None:
            close()

    def operation(self, request):
        if not isinstance(request, dict):
            raise ValueError("Each JSONL request must be an object")
        op = request.get("op")
        language = request.get("language")
        if language is not None and language not in ("zh", "en"):
            raise ValueError("language must be zh or en; it is never inferred from text")
        if op == "whole":
            return self.whole(request, language)
        if op not in ("begin_message", "append_text", "end_session"):
            raise ValueError("op must be whole, begin_message, append_text, or end_session")
        session_id = request.get("session")
        if not isinstance(session_id, str) or not session_id or len(session_id) > 256:
            raise ValueError("Streaming operations require a nonempty session string of at most 256 characters")
        if op == "end_session":
            if session_id not in self.sessions:
                raise ValueError("Unknown session")
            del self.sessions[session_id]
            return {"event": "session_ended", "session": session_id, "generated_tokens": 0}
        if not isinstance(request.get("text", ""), str):
            raise ValueError("text must be a string")
        engine_before = self._engine_stats()
        if op == "begin_message":
            if request.get("role") not in ROLES:
                raise ValueError("begin_message requires a user or assistant role")
            new = session_id not in self.sessions
            if new and len(self.sessions) >= self.args.max_sessions:
                raise ValueError("Session limit reached; end a session before creating another")
            entry = self.sessions.get(session_id) or {"session": self.runtime.new_session(), "language": None}
            result = entry["session"].begin_message(request["role"], request.get("text", ""))
            # A new message has no assumed language when the caller omits it.
            entry["language"] = language
            self.sessions[session_id] = entry
        else:
            if session_id not in self.sessions:
                raise ValueError("Unknown session; call begin_message first")
            if "text" not in request:
                raise ValueError("append_text requires text")
            entry = self.sessions[session_id]
            result = entry["session"].append_text(request["text"])
            if language is not None:
                entry["language"] = language
        language = entry["language"]
        seconds = result["wall_seconds"]
        result.update(event="classification", op=op, session=session_id, language=language,
                      engine_execution=self._request_execution(engine_before, result),
                      # Net token growth can be negative when BPE merges a suffix.
                      # Replay work is reported separately and never sold as new input.
                      observed_net_input_itps=result["net_new_tokens"] / seconds if seconds > 0 else None,
                      timing_scope="text tokenization + transactional cache restore/clone/replay + CPU-visible risk; includes cold calls; excludes JSON I/O")
        result.update(threshold_result(self.metadata["selection"], language, result["target_role"],
                                       result["risk_probabilities"], self.args.threshold, True))
        return result

    def whole(self, request, language):
        torch = self.torch
        messages = validate_messages(request.get("messages"))
        started = time.perf_counter()
        ids = self.tokenizer.encode(self.serialize(messages), add_special_tokens=False)
        if not ids or len(ids) > 8192:
            raise ValueError(f"Input has {len(ids)} tokens; allowed range is 1..8192; no truncation")
        model_limit = getattr(self.model.backbone.config, "max_position_embeddings", None)
        if isinstance(model_limit, int) and len(ids) > model_limit:
            raise ValueError("Input exceeds the configured position limit")
        tokenization_seconds = time.perf_counter() - started
        role = messages[-1]["role"]
        with torch.inference_mode():
            torch.cuda.synchronize()
            model_started = time.perf_counter()
            tokens = torch.tensor([ids], device="cuda", dtype=torch.long)
            output = self.model(tokens, use_cache=False)
            logits, _ = self.model.readout(output.last_hidden_state[:, -1], role)
            values = logits.float().softmax(-1)[0].cpu().tolist()
            torch.cuda.synchronize()
            model_seconds = time.perf_counter() - model_started
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Non-finite risk probabilities")
        wall_seconds = time.perf_counter() - started
        probabilities = dict(zip(RISK_LABELS, values))
        result = {"event": "classification", "op": "whole", "target_role": role,
                  "language": language, "risk_probabilities": probabilities,
                  "native_tokens": len(ids), "previous_native_tokens": 0,
                  "net_new_tokens": len(ids), "forward_tokens": len(ids), "forward_calls": 1,
                  "replay_tokens": 0, "rollback": False,
                  "tokenization_seconds": tokenization_seconds, "model_call_seconds": model_seconds,
                  "wall_seconds": wall_seconds, "observed_net_input_itps": len(ids) / wall_seconds,
                  "timing_scope": "tokenization + one complete model forward and target readout + CPU-visible risk; includes cold calls; excludes JSON I/O",
                  "generated_tokens": 0, "third_risk_class_validated": False,
                  "category_output_validated": False,
                  "system_message_stratum_validated": False,
                  "engine_execution": {"inference_engine": "eager_whole", "graph_calls": 0,
                                       "graph_forward_tokens": 0, "eager_calls": 1,
                                       "eager_forward_tokens": len(ids), "initialization_included": False}}
        result.update(threshold_result(self.metadata["selection"], language, role, probabilities,
                                       self.args.threshold, False))
        return result


def main(argv=None):
    args = parse_args(argv)
    if args.verify_only:
        bundle, manifest, selection = verify_bundle(args.bundle)
        print(json.dumps({"event": "bundle_verified", "bundle": str(bundle),
                          "candidate": selection["candidate"],
                          "inference_engine": INFERENCE_ENGINE,
                          "checkpoint_sha256": selection["checkpoint_sha256"],
                          "files": len(manifest["files"]), "model_calls": 0}))
        return
    torch, model, tokenizer, metadata = load_bundle(args.bundle)
    guard = JSONLGuard(args, torch, model, tokenizer, metadata)
    incoming = outgoing = None
    failed = False

    def emit(value):
        outgoing.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        outgoing.flush()

    def reject_non_json_constant(value):
        raise ValueError(f"Nonfinite constant {value} is not valid JSON")

    try:
        incoming = sys.stdin if str(args.input) == "-" else args.input.open(encoding="utf-8")
        outgoing = sys.stdout if str(args.output) == "-" else args.output.open("w", encoding="utf-8")
        emit({"event": "metadata", **{key: value for key, value in metadata.items() if key != "selection"},
              "inference_engine": guard.inference_engine, "engine_initialization": guard.engine_metadata,
              "max_input_tokens": 8192, "max_sessions": args.max_sessions,
              "risk_output": "three probabilities; only safe/unsafe received hard labels this round",
              "itps_accounting": "net native input-token growth / request wall time; replay tokens reported separately"})
        for line_number, line in enumerate(incoming, 1):
            if not line.strip():
                continue
            request = None
            try:
                request = json.loads(line, parse_constant=reject_non_json_constant)
                result = guard.operation(request)
                if "id" in request:
                    result["id"] = request["id"]
                emit(result)
            except (ValueError, TypeError, KeyError) as error:
                failed = True
                result = {"event": "error", "line": line_number,
                          "error_type": type(error).__name__, "message": str(error), "generated_tokens": 0}
                if isinstance(request, dict) and "id" in request:
                    result["id"] = request["id"]
                emit(result)
    finally:
        if incoming is not None and incoming is not sys.stdin:
            incoming.close()
        if outgoing is not None and outgoing is not sys.stdout:
            outgoing.close()
        guard.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
