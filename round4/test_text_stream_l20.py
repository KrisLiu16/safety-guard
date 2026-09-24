"""Exercise actual trained L20 caches through append-only text sessions."""
import gc
import hashlib
import json
import math
import statistics
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path[:0] = ['/work/validation', '/work/input', '/work/window', '/work/round4']
import torch
from infer_classifier import load_classifier, sha256_file
from text_stream_runtime import TextStreamRuntime, audit_cache_tensor_isolation, RISK_LABELS

OUT = Path('/work/output/round4_validation')
TRAINED = Path('/work/output/round4')


def cache_digest(cache):
    digest = hashlib.sha256()
    seen = set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif hasattr(value, '__dict__'):
            visit(vars(value))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            digest.update(str(value).encode())
    visit(cache)
    return digest.hexdigest()


@torch.inference_mode()
def oracle(model, session, result):
    ids = torch.tensor([session.token_ids], device='cuda')
    hidden = model(ids).last_hidden_state[:, -1]
    logits, _ = model.readout(hidden, result['target_role'])
    expected = logits.softmax(-1)[0].cpu().tolist()
    return max(abs(value - result['risk_probabilities'][label]) for label, value in zip(RISK_LABELS, expected))


def test_variant(name, variant):
    checkpoint = TRAINED / name / 'best.safetensors'
    setup = SimpleNamespace(base_root=Path('/work'), base_code_dir=Path('/work/input'),
                            window_code_dir=Path('/work/window'), memory_code_dir=Path('/work/round4'),
                            checkpoint=checkpoint)
    _, model, tok, _, _, _ = load_classifier(setup, variant)
    runtime = TextStreamRuntime(model, tok)
    errors = []
    records = []

    def checked(session, result, case):
        error = oracle(model, session, result)
        errors.append(error)
        records.append({'case': case, 'probability_error': error,
                        **{k: result[k] for k in ('native_tokens', 'net_new_tokens', 'forward_tokens',
                             'rollback', 'restore_position', 'fallback_full_replay', 'snapshot_positions')}})

    # A real native-BPE contraction across a snapshot boundary, followed by growth.
    contraction = runtime.new_session()
    first = contraction.begin_message('user', 'x ' * 187 + 'informatio')
    checked(contraction, first, 'contraction_before')
    middle = contraction.append_text('n')
    checked(contraction, middle, 'contraction_192_to_191')
    last = contraction.append_text('!')
    checked(contraction, last, 'contraction_grow_back')
    assert first['native_tokens'] == 192 and middle['native_tokens'] == 191 and last['native_tokens'] == 192

    letters = runtime.new_session()
    checked(letters, letters.begin_message('user', ''), 'empty_user_header')
    for character in '请解释中立讨论。information':
        checked(letters, letters.append_text(character), 'character_append')
    checked(letters, letters.begin_message('assistant', '可以结合上下文解释，不因词语出现就判定风险。'), 'assistant_role')
    checked(letters, letters.begin_message('user', '那英文定义呢？'), 'next_user_role')
    before = letters.token_ids
    no_change = letters.append_text('')
    assert no_change['forward_tokens'] == 0 and letters.token_ids == before

    background = ('这是一段公开阅读材料，内容需要结合上下文理解。 General public reading material.\n' * 110)
    long_session = runtime.new_session()
    checked(long_session, long_session.begin_message('user', background), 'long_prefill')
    assert len(long_session.token_ids) > 1024
    other = runtime.new_session()
    checked(other, other.begin_message('user', '另一场会话中的文字。'), 'isolated_session_start')
    before_storage = audit_cache_tensor_isolation(long_session)['total_tensor_storage_bytes']
    for text in ('补充', '一个', 'informatio', 'n', ' 的定义。', '\nNext sentence.'):
        checked(long_session, long_session.append_text(text), 'long_variable_append')
        checked(other, other.append_text('独立。'), 'interleaved_session')
    isolation = audit_cache_tensor_isolation(contraction, letters, long_session, other)

    # Reject overflow before changing any recurrent state or accepted text.
    before_ids = long_session.token_ids
    before_messages = long_session.messages
    before_digest = cache_digest(long_session._cache)
    try:
        long_session.append_text('超出长度限制的文字。' * 10000)
        raise AssertionError('Overflow should have been rejected')
    except ValueError:
        pass
    assert long_session.token_ids == before_ids and long_session.messages == before_messages
    assert cache_digest(long_session._cache) == before_digest

    # Mutate a real candidate cache by forwarding, then inject a failure before commit.
    normal_runner = long_session._runner
    def fail_after_forward(ids, cache):
        normal_runner(ids, cache)
        raise RuntimeError('intentional post-forward transaction failure')
    long_session._runner = fail_after_forward
    try:
        long_session.append_text('失败测试')
        raise AssertionError('Injected failure did not occur')
    except RuntimeError as error:
        assert 'intentional post-forward' in str(error)
    finally:
        long_session._runner = normal_runner
    assert long_session.token_ids == before_ids and long_session.messages == before_messages
    assert cache_digest(long_session._cache) == before_digest
    checked(long_session, long_session.append_text('恢复正常。'), 'retry_after_failure')

    # Warm model kernels first; then measure actual text processing, including
    # retokenization, snapshot copies and replay. Oracle calls are outside timing.
    chunks = [' 中立资料 public information。'] * 128
    long_session.append_text(chunks[0])
    # Warm the complete identical schedule in an independent session. A single
    # append cannot warm shapes introduced later by snapshot-boundary splitting.
    warm_session = runtime.new_session()
    for message in long_session.messages:
        warm_session.begin_message(message['role'], message['content'])
    for text in chunks:
        warm_session.append_text(text)
    del warm_session
    gc.collect()
    initial = len(long_session.token_ids)
    measurements = [long_session.append_text(text) for text in chunks]
    elapsed = sum(row['wall_seconds'] for row in measurements)
    net_tokens = len(long_session.token_ids) - initial
    latencies = sorted(row['wall_seconds'] * 1000 for row in measurements)
    after_storage = audit_cache_tensor_isolation(long_session)['total_tensor_storage_bytes']
    bounded = variant == 'full' or before_storage == after_storage
    checked(long_session, measurements[-1], 'benchmark_end')
    result = {
        'name': name, 'variant': variant, 'checkpoint_sha256': sha256_file(checkpoint),
        'max_probability_error': max(errors), 'tolerance': .03,
        'all_text_parity_pass': all(error < .03 for error in errors),
        'cache_isolation': isolation, 'overflow_transaction_pass': True, 'failed_forward_transaction_pass': True,
        'bounded_session_state_pass': bounded if variant != 'full' else None,
        'session_state_bytes_before': before_storage, 'session_state_bytes_after': after_storage,
        'pass': all(error < .03 for error in errors) and bounded,
        'cases': records,
        'text_stream_performance': {
            'initial_native_tokens': initial, 'accepted_appends': len(measurements), 'net_new_native_tokens': net_tokens,
            'actual_forward_tokens': sum(row['forward_tokens'] for row in measurements),
            'replay_tokens': sum(row['replay_tokens'] for row in measurements),
            'rollback_events': sum(row['rollback'] for row in measurements),
            'seconds': elapsed, 'itps_net_native_input': net_tokens / elapsed,
            'classifications_per_second': len(measurements) / elapsed,
            'input_utf8_bytes_per_second': sum(len(text.encode()) for text in chunks) / elapsed,
            'p50_ms': statistics.median(latencies), 'p95_ms': latencies[math.ceil(.95 * len(latencies)) - 1],
            'scope': 'L20 warmed text append API; includes tokenization, clones, replay and CPU-visible probabilities; '
                     'no HTTP or input-arrival waiting; denominator counts accepted net tokens, not replay tokens'},
        'generated_tokens': 0,
    }
    del runtime, model, contraction, letters, long_session, other
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    assert torch.cuda.device_count() == 1 and 'L20' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((TRAINED / 'final_summary.json').read_text())
    results = {}
    for name, variant in [(name, name) for name in ('full', 'window', 'memory')] + [('classification_rl', summary['rl_base_variant'])]:
        try:
            results[name] = test_variant(name, variant)
        except Exception as error:
            results[name] = {'pass': False, 'error_type': type(error).__name__, 'error': str(error)}
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
        (OUT / 'text_stream_audit.json').write_text(json.dumps({
            'status': 'completed' if len(results) == 4 else 'running',
            'all_candidates_pass': all(result['pass'] for result in results.values()),
            'results': results, 'generated_tokens': 0}, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'text_stream_audit': name, 'pass': results[name]['pass']}), flush=True)


if __name__ == '__main__':
    main()
