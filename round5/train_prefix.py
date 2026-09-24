"""Fixed W512 risk-only prefix post-training and binary classification-action RL.

Importing this module and --verify-data-only never load a neural model. The
formal run requires a previously completed modern native-tokenizer proof.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import stream_metrics

SEED = 20260925
EPOCHS, EFFECTIVE_BATCH, MICROBATCH = 2, 16, 4
START_SHA = 'bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2'
START_PATH = Path('/work/output/round4/window/best.safetensors')
COUNTS = {'train': 33311, 'calibration': 900, 'dev': 1200}
VERSIONS = {'transformers': '5.17.0', 'tokenizers': '0.23.2'}
WEIGHTS = {'public_source_reference': 1.0, 'synthetic_weak': .2,
           'rubric_benign': .8, 'synthetic_context_weak': .4}
RL_TIERS = ('public_source_reference', 'rubric_benign', 'synthetic_weak')
ASSETS = ('tokenizer.json', 'tokenizer_config.json', 'config.json', 'vocab.json',
          'merges.txt', 'added_tokens.json', 'special_tokens_map.json', 'chat_template.jinja')
PROOF_VERSION = 'prefix-all-view-native-ids-v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_rows(path, rows):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(canonical(row) + '\n')
    temporary.replace(path)


def serialize(messages):
    return '\n\n'.join(message['role'].upper() + ':\n' + message['content'] for message in messages)


def view_inputs(row, split):
    """Original IDs are never replaced; every derived view has its own text."""
    yield 'original', serialize(row['messages']), row['ids']
    if split == 'train':
        messages = [dict(message) for message in row['messages']]
        messages[-1]['content'] += row['augmentation']['suffix']
        yield 'augmentation', serialize(messages), row['augmentation']['ids']
    else:
        for view in row['text_cut_views']:
            messages = [dict(message) for message in row['messages']]
            messages[-1]['content'] = messages[-1]['content'][:view['cut_chars']]
            yield 'cut:' + str(view['cut_chars']), serialize(messages), view['ids']


def valid_ids(ids):
    return (isinstance(ids, list) and 0 < len(ids) <= 8192
            and all(type(token) is int and token >= 0 for token in ids))


def load_data(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['version'] != 'prefix-supervision-v2' or manifest['counts'] != COUNTS:
        raise ValueError('The fixed prefix-v2 dataset counts/version changed')
    data, families, sample_ids = {}, {}, set()
    for split in COUNTS:
        path = root / (split + '.jsonl')
        if sha(path) != manifest['output_hashes'][split]:
            raise ValueError('Frozen data hash mismatch: ' + split)
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != COUNTS[split]:
            raise ValueError('Frozen record count mismatch: ' + split)
        families[split] = set()
        for row in rows:
            if (row['sample_id'] in sample_ids or row['base_split'] != split
                    or row['prefix_training_version'] != manifest['version']):
                raise ValueError('Duplicate ID or mismatched split/version')
            sample_ids.add(row['sample_id'])
            families[split].add(row['family'])
            if (row['target_role'] not in ('user', 'assistant')
                    or row['messages'][-1]['role'] != row['target_role']
                    or row['source_label'] not in ('safe', 'unsafe')
                    or row['weight'] != WEIGHTS[row['label_tier']]):
                raise ValueError('Invalid role, label or fixed source weight')
            if not valid_ids(row['ids']) or ids_sha(row['ids']) != row['original_ids_sha256']:
                raise ValueError('Original IDs invalid or modified')
            positions = row['target_token_positions']
            if (not positions or positions != sorted(set(positions))
                    or positions[-1] != len(row['ids']) - 1
                    or any(type(p) is not int or p < 0 or p >= len(row['ids']) for p in positions)):
                raise ValueError('Invalid target token positions')
            if split == 'train':
                for anchor in row['anchors']:
                    if (row['source_label'] != 'safe' or anchor['label'] != 0
                            or anchor['confidence'] != .1
                            or anchor['label_origin'] != 'whole_safe_inherited_weak'
                            or anchor['token_end_exclusive'] - 1 not in positions
                            or not 1 <= anchor['token_end_exclusive'] < len(row['ids'])):
                        raise ValueError('Only explicitly weak safe interior anchors are permitted')
                aug = row['augmentation']
                if (aug['kind'] != 'neutral_suffix_after_complete_target'
                        or aug['original_endpoint_reused'] is not False
                        or aug.get('complete_target_verified_by') != 'exact decoded causal prefix starts with original serialized text'
                        or aug['epoch_phase'] not in (0, 1) or not aug['anchors']
                        or aug['suffix'] not in manifest['suffixes']
                        or not valid_ids(aug['ids']) or ids_sha(aug['ids']) != aug['ids_sha256']):
                    raise ValueError('Invalid complete-target augmentation')
                for anchor in aug['anchors']:
                    if (anchor['label'] != int(row['source_label'] == 'unsafe')
                            or anchor['confidence'] != 1.0
                            or anchor['label_origin'] != 'complete_target_plus_neutral_suffix'
                            or not 1 <= anchor['token_end_exclusive'] <= len(aug['ids'])):
                        raise ValueError('Invalid complete suffix anchor')
            else:
                if row['label_tier'] != 'public_source_reference':
                    raise ValueError('Independent cal/dev source labels required')
                cuts = row['text_cut_views']
                if not cuts or len({view['cut_chars'] for view in cuts}) != len(cuts):
                    raise ValueError('Empty/duplicate text-cut schedule')
                for view in cuts:
                    if (not 1 <= view['cut_chars'] <= len(row['messages'][-1]['content'])
                            or not valid_ids(view['ids']) or ids_sha(view['ids']) != view['ids_sha256']):
                        raise ValueError('Invalid frozen text-cut view')
        data[split] = rows
    for left in families:
        for right in families:
            if left < right and families[left] & families[right]:
                raise ValueError('Family leakage across frozen splits')
    return manifest, data


def view_digest(data):
    digest, counts = hashlib.sha256(), collections.Counter()
    for split, rows in data.items():
        for row in rows:
            for kind, _, ids in view_inputs(row, split):
                digest.update((canonical([split, row['sample_id'], kind, ids]) + '\n').encode())
                counts[split + '/' + kind.split(':')[0]] += 1
    return digest.hexdigest(), dict(counts)


def asset_hashes(root):
    return {name: sha(root / name) if (root / name).is_file() else None for name in ASSETS}


def proof_binding(args, data):
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise RuntimeError('Actual modern tokenizer versions required: ' + canonical(VERSIONS))
    digest, counts = view_digest(data)
    return {'version': PROOF_VERSION, 'status': 'passed', 'library_versions': versions,
            'manifest_sha256': sha(args.data_root / 'manifest.json'),
            'split_sha256': {s: sha(args.data_root / (s + '.jsonl')) for s in COUNTS},
            'trainer_source_sha256': sha(Path(__file__)),
            'tokenizer_assets_sha256': asset_hashes(args.tokenizer_root),
            'all_saved_views_sha256': digest, 'view_counts': counts,
            'verified_views': sum(counts.values()), 'mismatched_views': 0,
            'verified_complete_target_anchors': sum(len(row['augmentation']['anchors']) for row in data['train']),
            'neural_model_loaded': False, 'neural_forward_calls': 0,
            'scope': 'all frozen original, augmentation and text-cut IDs; no replacement IDs'}


def check_encoding(tokenizer, text, expected):
    actual = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if actual != expected or not valid_ids(actual):
        raise ValueError('Actual modern native IDs differ from the frozen view; replacement forbidden')


def verify_data_only(args, data):
    if args.proof.exists():
        raise FileExistsError('Refusing to overwrite an existing tokenizer proof')
    binding = proof_binding(args, data)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_root, local_files_only=True)
    if tokenizer.backend_tokenizer.truncation is not None:
        raise ValueError('Native tokenizer must not have implicit truncation enabled')
    start = time.monotonic()
    checked = 0
    for split, rows in data.items():
        for row in rows:
            for kind, text, ids in view_inputs(row, split):
                try:
                    check_encoding(tokenizer, text, ids)
                except ValueError as error:
                    raise ValueError('Native proof mismatch: ' + split + '/' + row['sample_id'] + '/' + kind) from error
                checked += 1
            if split == 'train':
                original = serialize(row['messages'])
                for anchor in row['augmentation']['anchors']:
                    decoded = tokenizer.backend_tokenizer.decode(
                        row['augmentation']['ids'][:anchor['token_end_exclusive']], skip_special_tokens=False)
                    if not decoded.startswith(original):
                        raise ValueError('A supposedly complete suffix anchor omits original text: ' + row['sample_id'])
        print(canonical({'data_verification_split': split, 'verified_views': checked}), flush=True)
    if checked != binding['verified_views']:
        raise AssertionError('Proof coverage mismatch')
    binding.update(tokenizer_class=type(tokenizer).__name__, seconds=time.monotonic() - start)
    atomic_json(args.proof, binding)
    print(canonical({'proof': str(args.proof), 'sha256': sha(args.proof), 'verified_views': checked,
                     'neural_forward_calls': 0}), flush=True)


def verify_existing_proof(args, data):
    expected = proof_binding(args, data)
    actual = json.loads(args.proof.read_text())
    for name, value in expected.items():
        if actual.get(name) != value:
            raise ValueError('Frozen tokenizer proof binding mismatch: ' + name)
    return sha(args.proof)


def prefix_plan(row, epoch):
    original = row['anchors']
    augmented = row['augmentation']['anchors'] if epoch % 2 == row['augmentation']['epoch_phase'] else []
    return original, augmented, len(original) + len(augmented)


def epoch_groups(rows, epoch):
    # Same length-bucket/effective-batch grouping as round4, without a window curriculum.
    rng, groups = random.Random(SEED + epoch), []
    ordered = sorted(rows, key=lambda row: len(row['ids']))
    for start in range(0, len(ordered), 256):
        block = ordered[start:start + 256]
        rng.shuffle(block)
        groups.extend(block[i:i + EFFECTIVE_BATCH] for i in range(0, len(block), EFFECTIVE_BATCH))
    rng.shuffle(groups)
    return groups


def rl_pool(rows):
    # Sample records, never flatten prefixes into an over-weighted record pool.
    pool = [row for row in rows if row['label_tier'] in RL_TIERS]
    random.Random(SEED + 91).shuffle(pool)
    return pool


def import_training(args):
    # Do not import train_risk (which imports torch/model helpers) before proof validation.
    sys.path[:0] = [str(args.round4_code), '/work/input', '/work/window']
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'L20' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Neural execution requires one CUDA L20; CPU/MPS are forbidden')
    import train_risk as helper
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    random.seed(SEED)
    return torch, helper, load_file


def make_model(helper, load_file, checkpoint):
    backbone, tokenizer, _ = helper.load('qwen35')
    if getattr(backbone.config, 'num_hidden_layers', None) != 24:
        raise RuntimeError('All 24 original token-mixing layers must be retained')
    model = helper.Classifier(backbone).to('cuda')
    helper.configure(model, 'window', 512)
    state = load_file(str(checkpoint))
    model.load_state_dict(state, strict=True)
    del state
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def forward_hidden(helper, model, rows, pad):
    ids, mask = helper.batch(rows, pad)
    return model(ids, mask, use_cache=False).last_hidden_state


def finite_backward(torch, loss):
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite loss')
    loss.backward()
    return float(loss.detach())


def sft_update(torch, helper, model, tokenizer, optimizer, group, epoch):
    """Original and augmentation graphs are backwarded separately, then discarded."""
    F = torch.nn.functional
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, tokens, original_calls, augmentation_calls = 0., 0, 0, 0
    for start in range(0, len(group), MICROBATCH):
        rows = group[start:start + MICROBATCH]
        hidden = forward_hidden(helper, model, rows, tokenizer.pad_token_id)
        losses = []
        for index, row in enumerate(rows):
            original, _, count = prefix_plan(row, epoch)
            endpoint, _ = model.readout(hidden[index, len(row['ids']) - 1], row['target_role'])
            target = torch.tensor(int(row['source_label'] == 'unsafe'), device='cuda')
            loss = F.cross_entropy(endpoint, target) * row['weight']
            if original:
                positions = [anchor['token_end_exclusive'] - 1 for anchor in original]
                logits, _ = model.readout(hidden[index, positions], row['target_role'])
                labels = torch.tensor([anchor['label'] for anchor in original], device='cuda')
                confidence = torch.tensor([anchor['confidence'] for anchor in original], device='cuda')
                loss = loss + .5 * row['weight'] * (confidence * F.cross_entropy(logits, labels, reduction='none')).sum() / count
            losses.append(loss)
        total_loss += finite_backward(torch, torch.stack(losses).sum() / len(group))
        tokens += sum(len(row['ids']) for row in rows)
        original_calls += 1
        del hidden, losses, loss, endpoint
        augmented_rows = [row for row in rows if prefix_plan(row, epoch)[1]]
        if augmented_rows:
            views = [dict(row, ids=row['augmentation']['ids']) for row in augmented_rows]
            hidden = forward_hidden(helper, model, views, tokenizer.pad_token_id)
            losses = []
            for index, row in enumerate(augmented_rows):
                _, anchors, count = prefix_plan(row, epoch)
                logits, _ = model.readout(hidden[index, [a['token_end_exclusive'] - 1 for a in anchors]], row['target_role'])
                labels = torch.tensor([anchor['label'] for anchor in anchors], device='cuda')
                confidence = torch.tensor([anchor['confidence'] for anchor in anchors], device='cuda')
                losses.append(.5 * row['weight'] * (confidence * F.cross_entropy(logits, labels, reduction='none')).sum() / count)
            total_loss += finite_backward(torch, torch.stack(losses).sum() / len(group))
            tokens += sum(len(view['ids']) for view in views)
            augmentation_calls += 1
            del hidden, losses, logits
    grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    if not bool(torch.isfinite(grad)):
        raise RuntimeError('Nonfinite SFT gradient')
    optimizer.step()
    return {'loss': total_loss, 'gradient_norm': float(grad), 'input_tokens': tokens,
            'original_forward_calls': original_calls, 'augmentation_forward_calls': augmentation_calls}


def prepare_cut_jobs(rows, tokenizer):
    """Validate every cut first; deduplicate equivalent native input+role pairs."""
    jobs, index, plans = [], {}, {}
    for row in rows:
        plan = []
        for view, (_, text, ids) in zip(row['text_cut_views'], list(view_inputs(row, row['base_split']))[1:]):
            check_encoding(tokenizer, text, ids)
            if ids == row['ids']:
                plan.append({'cut_chars': view['cut_chars'], 'job_index': None})
                continue
            key = row['target_role'], tuple(ids)
            if key not in index:
                index[key] = len(jobs)
                jobs.append({'ids': ids, 'target_role': row['target_role'], 'job_index': len(jobs)})
            plan.append({'cut_chars': view['cut_chars'], 'job_index': index[key]})
        plans[row['sample_id']] = plan
    return jobs, plans


def observations(torch, helper, model, tokenizer, rows):
    results, by_id = [], {}
    jobs, cut_plans = prepare_cut_jobs(rows, tokenizer)
    cut_scores = {}
    ordered = sorted(rows, key=lambda item: (len(item['ids']), item['sample_id']))
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ordered), MICROBATCH):
            group = ordered[start:start + MICROBATCH]
            hidden = forward_hidden(helper, model, group, tokenizer.pad_token_id)
            for index, row in enumerate(group):
                positions = row['target_token_positions']
                logits, _ = model.readout(hidden[index, positions], row['target_role'])
                probabilities = logits.float().softmax(-1).cpu()
                end = probabilities[positions.index(len(row['ids']) - 1)].tolist()
                target = probabilities[:, 1].tolist()
                result = {key: row[key] for key in ('sample_id', 'family', 'language', 'target_role', 'source_label')} | {
                    'endpoint_probs': end, 'endpoint_p_unsafe': end[1], 'native_target_max_p_unsafe': max(target),
                    'target_token_positions': positions, 'native_target_p_unsafe': target}
                results.append(result)
                by_id[row['sample_id']] = result
        jobs.sort(key=lambda job: (len(job['ids']), job['job_index']))
        for start in range(0, len(jobs), MICROBATCH):
            group = jobs[start:start + MICROBATCH]
            hidden = forward_hidden(helper, model, group, tokenizer.pad_token_id)
            for index, job in enumerate(group):
                logits, _ = model.readout(hidden[index, len(job['ids']) - 1], job['target_role'])
                cut_scores[job['job_index']] = float(logits.float().softmax(-1)[1].cpu())
    for sample_id, plan in cut_plans.items():
        result = by_id[sample_id]
        cuts = [{'cut_chars': item['cut_chars'], 'p_unsafe': (result['endpoint_p_unsafe'] if item['job_index'] is None
                                                          else cut_scores[item['job_index']])} for item in plan]
        cut_max = max(item['p_unsafe'] for item in cuts)
        result.update(text_cut_observations=cuts, text_cut_max_p_unsafe=cut_max,
                      stream_max_p_unsafe=max(result['native_target_max_p_unsafe'], cut_max),
                      original_endpoint_reused_for_complete_cut=True)
    return results


def evaluate_checkpoint(torch, helper, load_file, checkpoint, data, out, baseline):
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = make_model(helper, load_file, checkpoint)
    model.eval().requires_grad_(False)
    calibration = observations(torch, helper, model, tokenizer, data['calibration'])
    dev = observations(torch, helper, model, tokenizer, data['dev'])
    result = stream_metrics.evaluate(calibration, dev, baseline)
    result.update(checkpoint_sha256=sha(checkpoint), evaluation_weights='exported_bfloat16_backbone',
                  generated_tokens=0, selection_read_test=False)
    atomic_rows(out / 'calibration_predictions.jsonl', calibration)
    atomic_rows(out / 'dev_predictions.jsonl', dev)
    atomic_json(out / 'metrics.json', result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def save_state(torch, model, optimizer, path, metadata):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(dict(metadata, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    python_rng=random.getstate()), temporary)
    temporary.replace(path)


def restore_state(torch, model, optimizer, path, binding, stage):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['binding'] != binding or state['stage'] != stage:
        raise ValueError('Resume state belongs to another frozen run/stage')
    model.load_state_dict(state.pop('model'), strict=True)
    optimizer.load_state_dict(state.pop('optimizer'))
    torch.set_rng_state(state.pop('torch_rng'))
    torch.cuda.set_rng_state(state.pop('cuda_rng'))
    random.setstate(state.pop('python_rng'))
    return state


def copy_checkpoint(source, destination):
    temporary = destination.with_name(destination.name + '.tmp')
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def reconcile_selected(selected, destination):
    """A crash between best-copy and atomic state-save must not silently mix them."""
    expected = selected['checkpoint_sha256']
    if destination.is_file() and sha(destination) == expected:
        return
    source = Path(selected['source_checkpoint'])
    if not source.is_file() or sha(source) != expected:
        raise ValueError('Selected checkpoint cannot be recovered from its frozen source')
    copy_checkpoint(source, destination)


def trim_uncommitted_log(path, completed):
    if path.exists():
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        atomic_rows(path, [row for row in rows if row['step'] <= completed])


def completed_summary(out, binding):
    path = out / 'summary.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result['binding'] != binding or result['checkpoint_sha256'] != sha(out / 'best.safetensors'):
        raise ValueError('Completed stage summary/checkpoint belongs to different frozen inputs')
    return result


def choose(candidate, incumbent, checkpoint, best_path, stage, step):
    if stream_metrics.is_better(candidate, incumbent):
        copy_checkpoint(checkpoint, best_path)
        return candidate, {'stage': stage, 'step': step, 'checkpoint_sha256': sha(best_path),
                           'source_checkpoint': str(checkpoint)}
    return incumbent, None


def train_sft(torch, helper, load_file, args, data, binding):
    out = args.output / 'sft'
    out.mkdir(parents=True, exist_ok=True)
    complete = completed_summary(out, binding)
    if complete:
        return complete
    model, tokenizer = make_model(helper, load_file, START_PATH)
    helper.training_mode(model)
    backbone = [p for name, p in model.named_parameters() if name.startswith('backbone.') and p.requires_grad]
    heads = [p for name, p in model.named_parameters() if name.startswith('heads.') and p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': 8e-6, 'base_lr': 8e-6},
                                  {'params': heads, 'lr': 5e-5, 'base_lr': 5e-5}], weight_decay=.01, eps=1e-6)
    state_path, best_path = out / 'latest_training_state.pt', out / 'best.safetensors'
    groups = [(epoch, group) for epoch in range(EPOCHS) for group in epoch_groups(data['train'], epoch)]
    total = len(groups)
    if args.resume and state_path.exists():
        state = restore_state(torch, model, optimizer, state_path, binding, 'sft')
        completed, baseline, best, selected = state['step'], state['baseline'], state['best'], state['selected']
        reconcile_selected(selected, best_path)
        trim_uncommitted_log(out / 'losses.jsonl', completed)
    else:
        if best_path.exists():
            raise FileExistsError('Partial SFT without resumable state; use a fresh output directory')
        copy_checkpoint(START_PATH, best_path)
        initial = evaluate_checkpoint(torch, helper, load_file, best_path, data, out / 'evaluation_0', None)
        baseline, best, selected, completed = initial['whole']['macro_recall'], initial, {
            'stage': 'initial', 'step': 0, 'checkpoint_sha256': START_SHA, 'source_checkpoint': str(START_PATH)}, 0
        save_state(torch, model, optimizer, state_path,
                   {'binding': binding, 'stage': 'sft', 'step': 0, 'baseline': baseline, 'best': best, 'selected': selected})
    start = time.monotonic()
    with (out / 'losses.jsonl').open('a') as log:
        for step, (epoch, group) in enumerate(groups, 1):
            if step <= completed:
                continue
            fraction = step / total
            scale = min(1., step / 64) * (.1 + .9 * .5 * (1 + math.cos(math.pi * fraction)))
            for spec in optimizer.param_groups:
                spec['lr'] = spec['base_lr'] * scale
            record = sft_update(torch, helper, model, tokenizer, optimizer, group, epoch)
            record.update(step=step, steps=total, epoch=epoch, window=512, seconds=time.monotonic() - start)
            log.write(canonical(record) + '\n')
            epoch_end = step == total or groups[step][0] != epoch
            if step % 64 == 0 or epoch_end:
                log.flush()
                print(canonical({'stage': 'sft', **record}), flush=True)
            if epoch_end:
                checkpoint = out / ('epoch_' + str(epoch + 1) + '.safetensors')
                helper.export(model, checkpoint)
                measured = evaluate_checkpoint(torch, helper, load_file, checkpoint, data, out / ('evaluation_' + str(epoch + 1)), baseline)
                best, changed = choose(measured, best, checkpoint, best_path, 'sft', step)
                if changed:
                    selected = changed
            if step % 256 == 0 or epoch_end:
                save_state(torch, model, optimizer, state_path,
                           {'binding': binding, 'stage': 'sft', 'step': step, 'baseline': baseline, 'best': best, 'selected': selected})
    summary = {'status': 'completed', 'stage': 'sft', 'steps': total, 'epochs': EPOCHS,
               'baseline_whole_macro_recall': baseline, 'best_metrics': best, 'selected': selected,
               'checkpoint_sha256': sha(best_path), 'eligible_candidate_found': best['eligible'],
               'promoted_from_initial': selected['stage'] != 'initial', 'binding': binding}
    atomic_json(out / 'summary.json', summary)
    del model, optimizer, backbone, heads
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def rl_terms(torch, logits, reference_logits, label):
    """Four binary actions per trusted anchor; exact own-policy reward baseline."""
    F = torch.nn.functional
    pu = logits.softmax(-1)[:, 1].clamp(1e-5, 1 - 1e-5)
    pref = reference_logits.softmax(-1)[:, 1].clamp(1e-5, 1 - 1e-5)
    y = torch.full((len(logits),), label, device='cuda', dtype=torch.long)
    actions = torch.bernoulli(pu.detach()[:, None].expand(-1, 4))
    r0, r1 = torch.where(y == 0, 1., -4.), torch.where(y == 1, 1., -3.)
    rewards = torch.where(actions.bool(), r1[:, None], r0[:, None])
    baseline = (1 - pu.detach()) * r0 + pu.detach() * r1
    logp = actions * pu[:, None].log() + (1 - actions) * (1 - pu[:, None]).log()
    policy = -((rewards - baseline[:, None]) * logp).mean(1)
    kl = pu * (pu.log() - pref.log()) + (1 - pu) * ((1 - pu).log() - (1 - pref).log())
    ce = F.cross_entropy(logits, y, reduction='none')
    trace = {'binary_action_p_unsafe': pu.detach().cpu().tolist(),
             'reference_binary_p_unsafe': pref.detach().cpu().tolist(),
             'actions': actions.detach().to(dtype=torch.int64).cpu().tolist(),
             'rewards': rewards.detach().cpu().tolist(),
             'exact_reward_baseline': baseline.detach().cpu().tolist(),
             'policy_loss': policy.detach().cpu().tolist(), 'binary_kl': kl.detach().cpu().tolist(),
             'three_class_ce': ce.detach().cpu().tolist()}
    return policy + .1 * kl + .2 * ce, rewards.mean(1), policy, kl, trace


def rl_update(torch, helper, model, reference, tokenizer, optimizer, rows):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = collections.Counter()
    traces = []
    for start in range(0, len(rows), MICROBATCH):
        group = rows[start:start + MICROBATCH]
        # Separate backward calls release the original graph before augmented views.
        for kind in ('original', 'augmentation'):
            views = group if kind == 'original' else [dict(row, ids=row['augmentation']['ids']) for row in group]
            hidden = forward_hidden(helper, model, views, tokenizer.pad_token_id)
            with torch.no_grad():
                reference_hidden = forward_hidden(helper, reference, views, tokenizer.pad_token_id)
            losses = []
            for index, row in enumerate(group):
                positions = ([len(row['ids']) - 1] if kind == 'original' else
                             [a['token_end_exclusive'] - 1 for a in row['augmentation']['anchors']])
                logits, _ = model.readout(hidden[index, positions], row['target_role'])
                with torch.no_grad():
                    reference_logits, _ = reference.readout(reference_hidden[index, positions], row['target_role'])
                terms, reward, policy, kl, trace = rl_terms(torch, logits, reference_logits, int(row['source_label'] == 'unsafe'))
                # Every record contributes its mean over original+complete-suffix anchors.
                scale = row['weight'] / ((1 + len(row['augmentation']['anchors'])) * len(rows))
                losses.append(terms.sum() * scale)
                totals['weighted_reward'] += float(reward.sum().detach()) * scale
                totals['policy_loss'] += float(policy.sum().detach()) * scale
                totals['reference_policy_kl'] += float(kl.sum().detach()) * scale
                traces.append({'sample_id': row['sample_id'], 'label_tier': row['label_tier'],
                               'source_label': row['source_label'], 'target_role': row['target_role'],
                               'view_kind': kind, 'token_end_exclusive': [position + 1 for position in positions],
                               'source_weight': row['weight'], 'record_anchor_count': 1 + len(row['augmentation']['anchors']),
                               'loss_scale': scale, **trace})
            totals['loss'] += finite_backward(torch, torch.stack(losses).sum())
            totals['input_tokens'] += sum(len(view['ids']) for view in views)
            totals['reference_input_tokens'] += sum(len(view['ids']) for view in views)
            del hidden, reference_hidden, losses, terms, logits
    grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    if not bool(torch.isfinite(grad)):
        raise RuntimeError('Nonfinite classification RL gradient')
    optimizer.step()
    return dict(totals, gradient_norm=float(grad), generated_tokens=0, action_traces=traces)


def train_rl(torch, helper, load_file, args, data, binding, sft):
    out = args.output / 'classification_rl'
    out.mkdir(parents=True, exist_ok=True)
    complete = completed_summary(out, binding)
    if complete:
        return complete
    source = args.output / 'sft/best.safetensors'
    if sha(source) != sft['checkpoint_sha256']:
        raise ValueError('Selected SFT reference checkpoint was modified')
    reference, tokenizer = make_model(helper, load_file, source)
    reference.eval().requires_grad_(False)
    model, _ = make_model(helper, load_file, source)
    helper.training_mode(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-7, weight_decay=0., eps=1e-6)
    pool = rl_pool(data['train'])
    if not pool:
        raise ValueError('No rubric-eligible complete RL anchors')
    state_path, best_path = out / 'latest_training_state.pt', out / 'best.safetensors'
    baseline = sft['baseline_whole_macro_recall']
    if args.resume and state_path.exists():
        state = restore_state(torch, model, optimizer, state_path, binding, 'classification_rl')
        completed, best, selected = state['step'], state['best'], state['selected']
        reconcile_selected(selected, best_path)
        trim_uncommitted_log(out / 'trajectory_metrics.jsonl', completed)
    else:
        if best_path.exists():
            raise FileExistsError('Partial RL without resumable state; use a fresh output directory')
        copy_checkpoint(source, best_path)
        completed, best, selected = 0, sft['best_metrics'], {
            'stage': 'sft_selected', 'step': 0, 'checkpoint_sha256': sha(source), 'source_checkpoint': str(source)}
        torch.manual_seed(SEED + 91)
        save_state(torch, model, optimizer, state_path,
                   {'binding': binding, 'stage': 'classification_rl', 'step': 0, 'best': best, 'selected': selected})
    with (out / 'trajectory_metrics.jsonl').open('a') as log:
        for step in range(completed + 1, 257):
            rows = [pool[((step - 1) * 8 + index) % len(pool)] for index in range(8)]
            record = rl_update(torch, helper, model, reference, tokenizer, optimizer, rows)
            record.update(step=step, window=512, original_endpoints=len(rows),
                          complete_suffix_anchors=sum(len(row['augmentation']['anchors']) for row in rows))
            log.write(canonical(record) + '\n')
            if step % 64 == 0:
                log.flush()
                checkpoint = out / ('step_' + str(step) + '.safetensors')
                helper.export(model, checkpoint)
                measured = evaluate_checkpoint(torch, helper, load_file, checkpoint, data, out / ('evaluation_' + str(step)), baseline)
                best, changed = choose(measured, best, checkpoint, best_path, 'classification_rl', step)
                if changed:
                    selected = changed
                save_state(torch, model, optimizer, state_path,
                           {'binding': binding, 'stage': 'classification_rl', 'step': step, 'best': best, 'selected': selected})
                progress = {name: value for name, value in record.items() if name != 'action_traces'}
                print(canonical({'stage': 'classification_rl', **progress, 'selected': selected}), flush=True)
    summary = {'status': 'completed', 'stage': 'classification_rl', 'updates': 256,
               'baseline_whole_macro_recall': baseline, 'best_metrics': best, 'selected': selected,
               'checkpoint_sha256': sha(best_path), 'reference_checkpoint_sha256': sft['checkpoint_sha256'],
               'eligible_candidate_found': best['eligible'], 'promoted_from_sft': selected['stage'] == 'classification_rl',
               'reward': {'correct': 1, 'false_positive': -3, 'false_negative': -4},
               'actions_per_input': 4, 'natural_safe_interior_hard_rewards': 0, 'generated_tokens': 0,
               'rl_pool_size': len(pool), 'binding': binding}
    atomic_json(out / 'summary.json', summary)
    del model, reference, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/work/round5/data/prefix_v2'))
    parser.add_argument('--tokenizer-root', type=Path, default=Path('/work/models/qwen35'))
    parser.add_argument('--proof', type=Path, default=Path('/work/round5/native_tokenizer_proof.json'))
    parser.add_argument('--round4-code', type=Path, default=Path('/work/round4'))
    parser.add_argument('--output', type=Path, default=Path('/work/output/round5/prefix_v2'))
    parser.add_argument('--verify-data-only', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.verify_data_only and (args.smoke_only or args.resume):
        parser.error('CPU verification is a separate mode')
    _, data = load_data(args.data_root)
    if args.verify_data_only:
        verify_data_only(args, data)
        return
    proof_sha = verify_existing_proof(args, data)
    if args.tokenizer_root.resolve() != Path('/work/models/qwen35'):
        raise ValueError('Neural helper model assets are fixed at /work/models/qwen35')
    if not args.output.resolve().is_relative_to(Path('/work/output/round5')):
        raise ValueError('All training and smoke outputs must stay inside the new /work/output/round5 directory')
    if args.smoke_only and (args.resume or args.output.resolve() == Path('/work/output/round5/prefix_v2')):
        raise ValueError('Smoke requires a separate explicit output directory and cannot resume')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Refusing to overwrite an existing output; use --resume for an interrupted formal run')
    if sha(START_PATH) != START_SHA:
        raise ValueError('Fixed window/SFT starting checkpoint SHA mismatch')
    torch, helper, load_file = import_training(args)
    binding = {'proof_sha256': proof_sha, 'manifest_sha256': sha(args.data_root / 'manifest.json'),
               'trainer_sha256': sha(Path(__file__)), 'metrics_sha256': sha(Path(stream_metrics.__file__)),
               'round4_helper_sha256': sha(Path(helper.__file__)), 'initial_checkpoint_sha256': START_SHA,
               'seed': SEED, 'window': 512, 'effective_batch': EFFECTIVE_BATCH, 'microbatch': MICROBATCH}
    args.output.mkdir(parents=True, exist_ok=True)
    setup_path = args.output / 'run_spec.json'
    if setup_path.exists():
        existing = json.loads(setup_path.read_text())
        if existing['binding'] != binding or existing.get('smoke_only'):
            raise ValueError('Resume run specification differs')
    else:
        atomic_json(setup_path, {'binding': binding, 'epochs': 2, 'records_per_epoch': 33311,
                    'architecture': '24 original layers, fixed W512; no window curriculum',
                    'category_parameters_frozen': True, 'risk_and_backbone_parameters_trainable': True,
                    'sft_prefix_weight': .5, 'normalization': 'valid original+active augmentation anchor count, not confidence sum',
                    'sft_evaluations': [0, 1, 2], 'rl_evaluations': [64, 128, 192, 256],
                    'rl_lr': 5e-7, 'rl_batch': 8, 'rl_microbatch': 4, 'rl_allowed_tiers': RL_TIERS,
                    'rl_anchor_sampling': '8 fixed-seed source records; mean over each record original endpoint and complete suffix anchors before source weighting',
                    'rl_not_episode_stopping_reward': True,
                    'reference_policy': 'selected own SFT checkpoint; no semantic teacher',
                    'selection_uses_test': False, 'generated_tokens': 0, 'smoke_only': args.smoke_only})
    if args.smoke_only:
        model, tokenizer = make_model(helper, load_file, START_PATH)
        helper.training_mode(model)
        optimizer = torch.optim.AdamW([{'params': [p for n, p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad], 'lr': 8e-6},
                                      {'params': [p for n, p in model.named_parameters() if n.startswith('heads.') and p.requires_grad], 'lr': 5e-5}], weight_decay=.01, eps=1e-6)
        record = sft_update(torch, helper, model, tokenizer, optimizer, epoch_groups(data['train'], 0)[0], 0)
        atomic_json(args.output / 'smoke.json', {'status': 'completed', 'updates': 1, 'binding': binding, **record})
        return
    sft = train_sft(torch, helper, load_file, args, data, binding)
    rl = train_rl(torch, helper, load_file, args, data, binding, sft)
    atomic_json(args.output / 'summary.json', {'status': 'completed', 'sft': sft, 'classification_rl': rl,
                'final_checkpoint': str(args.output / 'classification_rl/best.safetensors'),
                'final_checkpoint_sha256': rl['checkpoint_sha256'], 'production_approval': False,
                'category_validated': False, 'third_risk_class_hard_supervised': False,
                'selection_read_test': False, 'generated_tokens': 0, 'binding': binding})


if __name__ == '__main__':
    main()
