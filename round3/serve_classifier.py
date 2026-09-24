"""Local-only HTTP adapter for complete and streaming C1 classifications."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

from classifier_runtime import (PrefixClassifier, classify_full,
                                load_baseline_classifier, load_stage0_classifier)


class ClassificationServer(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True

    def __init__(self, address, model, tokenizer, version, max_sessions=128,
                 ttl_seconds=1800, training_status="stage0_teacher_distilled_unaligned"):
        super().__init__(address, Handler)
        self.model, self.tokenizer, self.version = model, tokenizer, version
        self.training_status = training_status
        self.max_sessions, self.ttl_seconds = max_sessions, ttl_seconds
        self.sessions = {}
        self.lock = threading.RLock()

    def prune(self):
        now = time.monotonic()
        for sid, state in list(self.sessions.items()):
            if now - state["updated"] > self.ttl_seconds:
                del self.sessions[sid]


class Handler(BaseHTTPRequestHandler):
    server: ClassificationServer

    def _reply(self, code, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 1_000_000:
            raise ValueError("Expected a JSON body of at most 1 MB")
        body = json.loads(self.rfile.read(size))
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object")
        return body

    def do_GET(self):
        if urlsplit(self.path).path != "/health":
            return self._reply(404, {"error": "not_found"})
        self._reply(200, {"ready": True, "model_version": self.server.version,
                          "training_status": self.server.training_status})

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            body = self._body()
            if path == "/classify":
                messages = body.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a nonempty list")
                with self.server.lock:
                    result = classify_full(self.server.model, self.server.tokenizer,
                                           self.server.version, messages,
                                           self.server.training_status)
                return self._reply(200, result)
            if path == "/sessions":
                history, role = body.get("history", []), body.get("target_role")
                if not isinstance(history, list) or role not in ("user", "assistant"):
                    raise ValueError("history must be a list and target_role user/assistant")
                if any(not isinstance(m, dict) or m.get("role") not in ("user", "assistant")
                       or not isinstance(m.get("content"), str) for m in history):
                    raise ValueError("Invalid history message")
                with self.server.lock:
                    self.server.prune()
                    if len(self.server.sessions) >= self.server.max_sessions:
                        return self._reply(429, {"error": "session_capacity"})
                    sid = secrets.token_urlsafe(18)
                    self.server.sessions[sid] = {
                        "history": history, "role": role, "content": "", "finalized": False,
                        "stream": PrefixClassifier(self.server.model, self.server.tokenizer,
                                                   self.server.version, role,
                                                   self.server.training_status),
                        "updated": time.monotonic(),
                    }
                return self._reply(201, {"session_id": sid, "model_version": self.server.version})
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "append":
                chunk, final = body.get("chunk"), body.get("final", False)
                if not isinstance(chunk, str) or not isinstance(final, bool):
                    raise ValueError("chunk must be text and final must be boolean")
                if not chunk and not final:
                    raise ValueError("Empty non-final chunk")
                with self.server.lock:
                    state = self.server.sessions.get(parts[1])
                    if state is None:
                        return self._reply(404, {"error": "unknown_session"})
                    if state["finalized"]:
                        return self._reply(409, {"error": "session_finalized"})
                    content = state["content"] + chunk
                    messages = state["history"] + [{"role": state["role"], "content": content}]
                    result = state["stream"].update_messages(messages, partial=not final)
                    state["content"], state["finalized"] = content, final
                    state["updated"] = time.monotonic()
                return self._reply(200, result)
            return self._reply(404, {"error": "not_found"})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return self._reply(400, {"error": "bad_request", "detail": str(exc)[:300]})

    def do_DELETE(self):
        parts = urlsplit(self.path).path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "sessions":
            return self._reply(404, {"error": "not_found"})
        with self.server.lock:
            existed = self.server.sessions.pop(parts[1], None) is not None
        self._reply(200 if existed else 404, {"deleted": existed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path,
                        help="Experimental Stage0 adapter; omit for original A0 baseline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.adapter:
        model, tokenizer, version = load_stage0_classifier(args.model, args.adapter)
        status = "stage0_teacher_distilled_unaligned"
    else:
        model, tokenizer, version = load_baseline_classifier(args.model)
        status = "pretrained_baseline_unaligned"
    server = ClassificationServer((args.host, args.port), model, tokenizer, version,
                                  training_status=status)
    print(json.dumps({"listening": f"http://{args.host}:{args.port}",
                      "model_version": version}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
