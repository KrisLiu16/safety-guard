"""Index Citizen Lab keyword observations as noncommercial, unlabeled candidates."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import unicodedata

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'research_sources/citizenlab/chat-censorship-1eeb5e6.tar.gz'
ARCHIVE_META = ROOT / 'research_sources/citizenlab/source_manifest.json'
BASE = ROOT / 'derived/expanded-20260923/master.jsonl'
OUTPUT = ROOT / 'research_nc/citizenlab-1eeb5e6'
CSV_PREFIXES = ('SVP/', 'TOM-Skype--Sina-UC/', 'LINE/translated-block-lists/',
                'wechat/', 'june-4/', 'coronavirus/', 'microsoft-bing/')
CSV_FIELDS = ('keyword', 'word', 'keyword_combination', 'keyword_added', 'person')
URL_RE = re.compile(r'^(?:https?://|www\.)', re.I)


def normalize(value):
    return unicodedata.normalize('NFKC', value.strip()).casefold()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def emit_jsonl(path, records):
    with path.open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def main():
    meta = json.loads(ARCHIVE_META.read_text())
    if digest(ARCHIVE) != meta['sha256']:
        raise ValueError('Citizen Lab source archive hash mismatch')
    if OUTPUT.exists():
        raise FileExistsError(f'Immutable output already exists: {OUTPUT}')
    base_keys = {json.loads(line)['canonical'] for line in BASE.open(encoding='utf-8')}
    terms = {}
    excluded = Counter()
    family_rows = Counter()
    file_catalog = []
    selected_files = 0

    def ingest(raw, family, path, position, origin=None):
        if not isinstance(raw, str):
            excluded['non_string'] += 1
            return
        key = normalize(raw)
        if not key:
            excluded['empty'] += 1
            return
        if '\ufffd' in key or len(key) > 128 or any(ord(ch) < 32 for ch in key):
            excluded['invalid_or_overlong'] += 1
            return
        family_rows[family] += 1
        row = terms.get(key)
        if row is None:
            row = {'canonical': key, 'display': raw.strip(), 'families': set(),
                   'source_examples': [], 'observations': 0, 'origin_examples': []}
            terms[key] = row
        row['observations'] += 1
        row['families'].add(family)
        examples = row['source_examples']
        if len(examples) < 3 and all(x['path'] != path for x in examples):
            examples.append({'path': path, 'position': position})
        if origin and len(row['origin_examples']) < 2 and origin not in row['origin_examples']:
            row['origin_examples'].append(origin)

    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = member.name.split('/', 1)[-1]
            family = None
            origin = None
            data = None
            entries = 0
            if path.startswith('open-source/blacklists/'):
                family = 'open_source_blacklists'
                data = archive.extractfile(member).read()
                lines = data.decode('utf-8-sig', errors='replace').splitlines()
                origin = lines[0] if lines else None
                for position, raw in enumerate(lines[2:], 3):
                    ingest(raw, family, path, position, origin)
                    entries += 1
            elif path.startswith('livestream/') and path.endswith('.plain'):
                family = 'livestream_sinashow' if 'sinashow-raw-data/' in path else 'livestream_yy'
                data = archive.extractfile(member).read()
                for position, raw in enumerate(data.decode('utf-8-sig', errors='replace').splitlines(), 1):
                    if family == 'livestream_sinashow':
                        raw = re.sub(r'^\s*\d+:\s*', '', raw)
                    ingest(raw, family, path, position)
                    entries += 1
            elif path.startswith('search/rules/') and path.endswith('.txt'):
                family = 'search_rules'
                data = archive.extractfile(member).read()
                for position, raw in enumerate(data.decode('utf-8-sig', errors='replace').splitlines(), 1):
                    ingest(raw, family, path, position)
                    entries += 1
            elif path in ('qqmail/censored.txt', 'olympics/illegalwords.txt'):
                family = 'qqmail_censored' if path.startswith('qqmail/') else 'olympics_blocklist'
                data = archive.extractfile(member).read()
                for position, raw in enumerate(data.decode('utf-8-sig', errors='replace').splitlines(), 1):
                    ingest(raw, family, path, position)
                    entries += 1
            elif path.startswith('chinese-games/') and path.lower().endswith('.txt'):
                family = 'chinese_games_text'
                data = archive.extractfile(member).read()
                for position, raw in enumerate(data.decode('utf-8-sig', errors='replace').splitlines(), 1):
                    if raw.strip() and not any(ch.isalnum() for ch in raw):
                        excluded['game_symbol_only'] += 1
                        continue
                    ingest(raw, family, path, position)
                    entries += 1
            elif (path.startswith('chinese-games/') and path.lower().endswith('.csv')
                  and 'regex' not in path.lower()):
                family = 'chinese_games_csv'
                data = archive.extractfile(member).read()
                text = data.decode('utf-8-sig', errors='replace')
                for position, values in enumerate(csv.reader(io.StringIO(text, newline='')), 1):
                    if not values:
                        continue
                    words = values[0].split('|') if 'sensitiveChatWords.csv' in path else values[:1]
                    for raw in words:
                        if raw.strip() and not any(ch.isalnum() for ch in raw):
                            excluded['game_symbol_only'] += 1
                            continue
                        ingest(raw, family, path, position)
                        entries += 1
            elif re.fullmatch(r'apple/(?:update-2022-02-27/)?(?:cn|hk|tw)\.txt', path):
                family = 'apple_chinese_engraving'
                data = archive.extractfile(member).read()
                for position, raw in enumerate(data.decode('utf-8-sig', errors='replace').splitlines(), 1):
                    ingest(raw, family, path, position)
                    entries += 1
            elif re.fullmatch(r'apple/siri-dialog/zh_(?:CN|HK|TW)\.csv', path):
                family = 'apple_chinese_siri'
                data = archive.extractfile(member).read()
                for position, values in enumerate(csv.reader(io.StringIO(data.decode('utf-8-sig', errors='replace'), newline='')), 1):
                    if values:
                        ingest(values[0], family, path, position)
                        entries += 1
            elif path.endswith('.csv') and path.startswith(CSV_PREFIXES):
                data = archive.extractfile(member).read()
                text = data.decode('utf-8-sig', errors='replace')
                reader = csv.DictReader(io.StringIO(text, newline=''))
                columns = {name.strip().casefold(): name for name in (reader.fieldnames or []) if name}
                field = next((columns[x] for x in CSV_FIELDS if x in columns), None)
                if field:
                    family = path.split('/', 1)[0] + '_keyword_csv'
                    for position, row in enumerate(reader, 2):
                        ingest(row.get(field), family, path, position)
                        entries += 1
            if family is not None:
                selected_files += 1
                file_catalog.append({'path': path, 'family': family, 'bytes': member.size,
                                     'sha256': hashlib.sha256(data).hexdigest(),
                                     'source_entries_seen': entries, 'origin': origin})

    OUTPUT.mkdir(parents=True)
    emit_jsonl(OUTPUT / 'source_file_catalog.jsonl', sorted(file_catalog, key=lambda x: x['path']))
    roles = Counter()
    candidates = []
    new_task_seeds = []
    combined = set(base_keys)
    for key in sorted(terms):
        item = terms[key]
        role = ('url_or_domain' if URL_RE.match(key) else
                'combination_candidate' if '+' in key else
                'pattern_candidate' if '*' in key and any(f.startswith('apple_') for f in item['families']) else
                'text')
        roles[role] += 1
        is_new = key not in base_keys
        combined.add(key)
        id_ = hashlib.sha256(key.encode('utf-8')).hexdigest()[:20]
        candidates.append({'id': id_, 'canonical': key, 'display': item['display'],
                           'role': role, 'families': sorted(item['families']),
                           'source_examples': item['source_examples'],
                           'origin_examples': item['origin_examples'],
                           'observations': item['observations'], 'new_vs_permissive_base': is_new,
                           'candidate_status': 'source_recorded_unlabeled',
                           'generation_eligible': role == 'text', 'auto_block': False,
                           'source_license': 'CC BY-NC-SA 4.0'})
        if is_new and role == 'text':
            task_key = 'lex-' + id_
            new_task_seeds.append({'task_key': task_key, 'origin_group_id': task_key,
                                   'word': item['display'], 'canonical': key,
                                   'source_examples': item['source_examples'],
                                   'generation_state': 'not_started', 'auto_block': False,
                                   'source_license': 'CC BY-NC-SA 4.0'})
    emit_jsonl(OUTPUT / 'candidate_terms.jsonl', candidates)
    emit_jsonl(OUTPUT / 'new_task_seeds.jsonl', new_task_seeds)
    (OUTPUT / 'candidate_keys.txt').write_text('\n'.join(sorted(terms)) + '\n', encoding='utf-8')
    (OUTPUT / 'combined_candidate_keys.txt').write_text('\n'.join(sorted(combined)) + '\n', encoding='utf-8')
    summary = {
        'source_id': meta['source_id'], 'revision': meta['revision'],
        'archive_sha256': meta['sha256'], 'archive_bytes': meta['bytes'],
        'source_license': 'CC BY-NC-SA 4.0',
        'permissive_base_candidates': len(base_keys), 'research_source_candidates': len(terms),
        'research_new_vs_permissive_base': len(combined) - len(base_keys),
        'combined_noncommercial_candidates': len(combined),
        'research_role_counts': dict(roles), 'new_text_task_seeds': len(new_task_seeds),
        'selected_source_files': selected_files, 'family_observations': dict(family_rows),
        'excluded_entries': dict(excluded),
        'output_hashes': {p.name: digest(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()},
        'semantics': 'Historical platform blocking observations and open-source blacklists; not policy labels or auto-block rules',
        'model_requests_submitted': 0,
    }
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('research_source_candidates',
                                             'research_new_vs_permissive_base',
                                             'combined_noncommercial_candidates',
                                             'new_text_task_seeds', 'selected_source_files')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
