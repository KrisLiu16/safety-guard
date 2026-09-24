"""JSONL whole-message and actual text-append inference for a Round5 bundle.

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

from standalone_model import INFERENCE_ENGINE, GRAPH_ENGINE, RISK_LABELS, ROLES, load_bundle, valid_threshold, verify_bundle


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--input", type=Path, default=Path("-"), help="UTF-8 JSONL, default stdin")
    parser.add_argument("--output", type=Path, default=Path("-"), help="UTF-8 JSONL, default stdout")
    parser.add_argument("--max-sessions", type=int, default=4)
    parser.add_argument("--inference-engine", choices=(INFERENCE_ENGINE, GRAPH_ENGINE), default=INFERENCE_ENGINE,
                        help="Default eager; window_cuda_graph is an explicit experimental engine without full text-state validation")
    parser.add_argument("--threshold", type=float, help="Explicit uncalibrated binary threshold override")
    parser.add_argument("--verify-only", action="store_true", help="Check files/hashes; no torch/model call")
    args = parser.parse_args(argv)
    if not 1 <= args.max_sessions <= 64:
        parser.error("--max-sessions must be 1..64; model block size is always fixed at 32")
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
    mode = 'stream' if streaming else 'whole'
    if explicit is not None:
        if not valid_threshold(explicit):
            raise ValueError('Explicit threshold must be finite and in [0,1]')
        value, kind = explicit, "explicit_uncalibrated"
    elif language is None:
        value, kind = None, "language_not_provided"
    else:
        value = selection["thresholds"][mode].get(f"{language}/{role}")
        kind = 'selected_checkpoint_' + mode + '_calibration' if value is not None else "stratum_not_calibrated"
    return {"target_binary_decision": None if value is None else (
                "unsafe" if probabilities["unsafe"] > value else "safe"),
            "threshold": {"kind": kind, "value": value,
                          'mode': mode, 'comparison': '>',
                          "stratum": None if language is None else f"{language}/{role}"},
            'calibration_scope': selection.get('calibration_scope', {}).get(mode),
            'arbitrary_bpe_schedule_fpr_guarantee': False, 'episode_stopping_policy': False,
            "safe_prefix_release_validated": False}


class JSONLGuard:
    def __init__(self, args, torch, model, tokenizer, metadata):
        from standalone_model import _load_module
        bundle = Path(metadata["bundle"])
        module = _load_module("text_stream_runtime", bundle / "text_stream_runtime.py")
        _load_module("graph_stream", bundle / "graph_stream.py")
        _load_module("canonical_block_engine", bundle / "canonical_block_engine.py")
        canonical = _load_module("canonical_text_runtime", bundle / "canonical_text_runtime.py")
        self.inference_engine = getattr(args, "inference_engine", INFERENCE_ENGINE)
        if metadata.get("inference_engine") != INFERENCE_ENGINE:
            raise ValueError("This entry point requires the canonical32 eager-default bundle")
        self.runtime = canonical.CanonicalTextRuntime(model, tokenizer, inference_engine=self.inference_engine,
                                                      pad_token_id=metadata['selection']['execution_contract']['pad_token_id'])
        self.engine_metadata = self.runtime.engine_metadata
        if self.engine_metadata['execution_contract'] != metadata['selection']['execution_contract']:
            self.runtime.close()
            raise ValueError('Runtime canonical execution differs from its calibrated contract')
        self.serialize, self.encode = module.serialize, canonical.encode
        self.args, self.torch, self.model, self.tokenizer = args, torch, model, tokenizer
        self.metadata, self.sessions = metadata, {}

    def _engine_stats(self):
        return self.runtime.stats()

    def _request_execution(self, before, result):
        # Local transaction accounting is valid even when other sessions exist.
        value = dict(result['engine_execution'])
        if (value['graph_calls'] + value['eager_calls'] != result['forward_calls']
                or value['forward_tokens'] != result['forward_tokens']
                or value['real_forward_tokens'] + value['padding_tokens'] != result['forward_tokens']):
            raise RuntimeError('Canonical transaction accounting mismatch')
        return dict(value, inference_engine=self.inference_engine)

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
                      timing_scope="canonical32 text tokenization + aligned cache restore/clone/replay + future padding + CPU-visible risk; input ITPS excludes replay and padding")
        result.update(threshold_result(self.metadata["selection"], language, result["target_role"],
                                       result["risk_probabilities"], self.args.threshold, True))
        return result

    def whole(self, request, language):
        messages = validate_messages(request.get('messages'))
        began = time.perf_counter()
        ids = self.encode(self.tokenizer, messages)
        token_seconds = time.perf_counter() - began
        observed = self.runtime.classify_prefixes(ids)
        role = messages[-1]['role']
        values = observed.pop('probabilities_by_role_all_tokens')
        probabilities = dict(zip(RISK_LABELS, values[role][-1]))
        wall = time.perf_counter() - began
        result = dict(observed, event='classification', op='whole', target_role=role,
                      risk_probabilities=probabilities, language=language,
                      previous_native_tokens=0, net_new_tokens=len(ids), replay_tokens=0, rollback=False,
                      tokenization_seconds=token_seconds, model_call_seconds=observed['wall_seconds'],
                      wall_seconds=wall, observed_net_input_itps=len(ids) / wall if wall else None,
                      timing_scope='canonical32 tokenization + fixed32 blocks and both all-position heads; input numerator excludes future padding',
                      third_risk_class_validated=False, category_output_validated=False,
                      system_message_stratum_validated=False, execution_definition='canonical32-v1')
        result['engine_execution'] = self._request_execution(None, result)
        result.update(threshold_result(self.metadata['selection'], language, role, probabilities,
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
