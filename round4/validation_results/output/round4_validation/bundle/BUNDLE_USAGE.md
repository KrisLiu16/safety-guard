# Portable direct-classifier bundle

The bundle contains the complete selected classifier weights, Qwen3.5 text configuration/tokenizer, the exact window/memory and text-stream runtime code, dependency versions, and SHA-256 checksums. It does not load original base-model weights or the initial H24 checkpoint. The ordinary model constructor initializes nonpersistent RoPE buffers before the complete trained state is copied into BF16 backbone / FP32 heads and new memory parameters.

Package on the worker after the fixed manifest has `status: research_candidate`:

```bash
/work/modern/bin/python /work/round4/package_model.py \
  --manifest /work/round4/MODEL_MANIFEST.json \
  --model-assets /work/models/qwen35 \
  --window-code-dir /work/window \
  --output /work/release/guard-round4
```

The output directory must not already exist. The default links the complete weight file on the same filesystem and otherwise copies it; `--copy-weights` always copies. Copies or archives of the resulting bundle require no training directories. Treat the bundle and source checkpoint as immutable. `--check-only` validates package inputs without loading a neural model.

Run with exactly one visible L20, the recorded `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` image, a C compiler (`gcc`, `libc6-dev`), and the bundled `requirements.txt`. For the CUDA PyTorch wheel, install `torch==2.7.1+cu128` from `https://download.pytorch.org/whl/cu128` before installing the remaining requirements. The loader checks dependency versions and reads tokenizer assets locally only. Model execution on CPU/MPS is disabled.

```bash
python /path/to/bundle/run_guard.py --verify-only
python /path/to/bundle/run_guard.py --input requests.jsonl --output predictions.jsonl
```

Example `requests.jsonl` (one JSON object per line):

```jsonl
{"id":"whole","op":"whole","language":"zh","messages":[{"role":"user","content":"请解释公开资料中这个词的历史背景。"}]}
{"id":"s1","op":"begin_message","session":"demo","role":"user","language":"zh","text":"请解释"}
{"id":"s2","op":"append_text","session":"demo","text":"公开资料中这个词的历史背景。"}
{"id":"s3","op":"begin_message","session":"demo","role":"assistant","language":"zh","text":"可以从公开史料出发，"}
{"id":"s4","op":"append_text","session":"demo","text":"区分事实记录与个人评价。"}
{"op":"end_session","session":"demo"}
```

`whole` is stateless. Each successful `begin_message` / `append_text` returns a classification of the current conversation through its final user or assistant message. `begin_message` starts another message within an existing session or creates the session; it does not reset history. Text append re-tokenizes the conversation, restores a complete recurrent-state snapshot if BPE changes a suffix, and replays the required input. Requests exceeding 8,192 tokens are rejected without truncation. Session state is transactional. The default live-session limit is four; end sessions to release their caches.

`native_tokens`, `net_new_tokens`, `forward_tokens` and `replay_tokens` distinguish new input from replay work. BPE merges can make net token growth zero or negative. Reported ITPS uses net input-token growth and request wall time, includes cold calls, and is not the warmed throughput benchmark. There are always zero generated output tokens.

Language is explicit, never guessed. Binary decisions use the selected checkpoint's recorded calibration threshold only for a matching language/role; otherwise probabilities are returned with a null binary decision. A new message without `language` clears the previous message's language. `--threshold` is an explicitly uncalibrated override. Whole-input calibration applied to stream prefixes is labelled unvalidated, and a low-risk prefix is not permission to release it.

Only safe/unsafe labels were directly supervised this round. The preserved third probability is not an independently validated third risk level, category outputs are not emitted, and the research bundle does not imply production approval. Packaging integrity checks do not replace the separate L20 parity and text-stream validation.
