"""Bounded post-training L20 validation, before report collection and GPU release."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/work/round4')
OUT = Path('/work/output/round4')
# Filled from the exact, separately versioned validation files before upload.
LOCK = ROOT / 'postrun_validation_files.json'
LOCK_SHA256 = 'bc716e2ef0a287b3aee2cf7369ad4acf133da67d5636665db313151fa486fd57'


def run_cli(name, variant, messages):
    directory = OUT / 'inference_smoke'
    directory.mkdir(exist_ok=True)
    request = directory / (name + '_request.json')
    request.write_text(json.dumps({'messages': messages}, ensure_ascii=False) + '\n')
    checkpoint = OUT / 'classification_rl/best.safetensors'
    results = {}
    for mode, chunk in (('whole', 0), ('stream', 8)):
        command = [sys.executable, str(ROOT / 'infer_classifier_v1.py'),
                   '--variant', 'classification_rl', '--checkpoint', str(checkpoint),
                   '--messages', str(request), '--chunk-tokens', str(chunk),
                   '--calibration-metrics', str(OUT / 'classification_rl/exported_dev_metrics.json'),
                   '--language', 'zh']
        stdout = directory / (name + '_' + mode + '.jsonl')
        with stdout.open('wb') as dest, (directory / (name + '_' + mode + '.stderr.log')).open('wb') as errors:
            result = subprocess.run(command, stdout=dest, stderr=errors, timeout=180)
        if result.returncode:
            raise RuntimeError(f'Inference CLI failed: {name}/{mode}, exit={result.returncode}; see persisted logs')
        records = []
        for line in stdout.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
        metadata = next(r for r in records if r.get('event') == 'metadata')
        complete = next(r for r in records if r.get('event') == 'complete')
        predictions = [r for r in records if r.get('event') == 'classification']
        final = next(r for r in predictions if r['final'])
        if metadata['variant'] != variant or complete['generated_tokens'] != 0:
            raise RuntimeError('CLI architecture or direct-classification contract mismatch')
        if len(predictions) != complete['classification_calls'] or final['prefix_tokens'] != complete['native_input_tokens']:
            raise RuntimeError('CLI did not classify every input chunk')
        expected_calls = (complete['native_input_tokens'] + chunk - 1) // chunk if chunk else 1
        if len(predictions) != expected_calls or sum(r['new_input_tokens'] for r in predictions) != complete['native_input_tokens']:
            raise RuntimeError('CLI chunk count or token accounting mismatch')
        previous = 0
        for row in predictions:
            if row['input_token_start'] != previous or row['input_token_end'] <= previous:
                raise RuntimeError('CLI skipped or repeated a token interval')
            previous = row['input_token_end']
        results[mode] = {'metadata': metadata, 'complete': complete, 'final': final}
    if results['whole']['metadata']['token_ids_sha256'] != results['stream']['metadata']['token_ids_sha256']:
        raise RuntimeError('Whole and streamed CLI inputs differ')
    errors = {
        role: max(abs(results['whole']['final']['risk_probabilities_by_head'][role][label]
                      - results['stream']['final']['risk_probabilities_by_head'][role][label])
                  for label in ('safe', 'unsafe', 'controversial'))
        for role in ('user', 'assistant')
    }
    return {'case': name, 'target_role': messages[-1]['role'], 'variant': variant,
            'checkpoint_sha256': results['whole']['metadata']['checkpoint_sha256'],
            'input_tokens': results['whole']['complete']['native_input_tokens'],
            'stream_classification_calls': results['stream']['complete']['classification_calls'],
            'probability_errors': errors, 'pass': max(errors.values()) < .03,
            'scope': 'CLI contract and numerical parity; this fixture is not a safety-quality benchmark'}


def main():
    locked_bytes = LOCK.read_bytes()
    if hashlib.sha256(locked_bytes).hexdigest() != LOCK_SHA256:
        raise RuntimeError('Post-training validation manifest checksum mismatch')
    spec = json.loads(locked_bytes)
    for path, expected in spec['files'].items():
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Post-training validation file checksum mismatch: {path}')
    subprocess.run([sys.executable, str(ROOT / 'postrun_long_stream_v1.py')], check=True, timeout=600)
    summary = json.loads((OUT / 'final_summary.json').read_text())
    variant = summary['rl_base_variant']
    cases = {
        'user': [{'role': 'user', 'content': '请中立地介绍维尼这个文学角色，并说明它与现实人物的区别。'}],
        'assistant': [{'role': 'user', 'content': '请帮我解释“敏感词”这三个字是什么意思。'},
                      {'role': 'assistant', 'content': '敏感词通常指在某种语境或平台规则中需要结合上下文审慎判断的词语。单独出现这些字，并不等于表达了有害意图。'}],
    }
    results = {}
    for name, messages in cases.items():
        try:
            results[name] = run_cli(name, variant, messages)
        except Exception as error:
            results[name] = {'pass': False, 'error_type': type(error).__name__, 'error': str(error)}
        (OUT / 'inference_cli_audit.json').write_text(json.dumps({
            'status': 'completed' if len(results) == len(cases) else 'running',
            'all_cases_pass': all(r['pass'] for r in results.values()), 'results': results,
            'generated_tokens': 0}, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'inference_cli_audit': name, 'pass': results[name]['pass']}), flush=True)
    if not all(r['pass'] for r in results.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
