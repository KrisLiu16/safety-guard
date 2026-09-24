"""Retrieve the verified full classifier through a CPU-only artifact Pod.

Metadata arrives with the validation report archive. This streams only the
large weight file and resumes a preserved partial download if interrupted.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from standalone_model import sha256_file, verify_bundle


def safe_bundle_path(root, relative):
    """Resolve a canonical manifest path without accepting symlink components."""
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or Path(relative).as_posix() != relative
            or any(part in ('.', '..') for part in Path(relative).parts)):
        raise ValueError('Invalid relative bundle path: ' + repr(relative))
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Bundle root must not be a symlink')
    root = root.resolve()
    target = root
    for part in Path(relative).parts:
        target = target / part
        if target.is_symlink():
            raise ValueError('Symlink bundle path is not allowed: ' + relative)
    if target == root or not target.resolve().is_relative_to(root):
        raise ValueError('Bundle path escapes its root: ' + relative)
    return target


def inspect_metadata(metadata):
    """Validate the whitelist and every small source file before any copying."""
    manifest_path = safe_bundle_path(metadata, 'BUNDLE_MANIFEST.json')
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('format') != 'round4-direct-classifier-bundle-v1':
        raise ValueError('Unsupported bundle manifest format')
    records = manifest.get('files')
    if not isinstance(records, dict) or 'classifier.safetensors' not in records:
        raise ValueError('Missing full classifier in bundle manifest')
    if 'BUNDLE_MANIFEST.json' in records:
        raise ValueError('The bundle manifest cannot list itself')
    sources = {}
    for relative, record in records.items():
        source = safe_bundle_path(metadata, relative)
        if (not isinstance(record, dict) or not isinstance(record.get('bytes'), int)
                or isinstance(record['bytes'], bool) or record['bytes'] < 0
                or not isinstance(record.get('sha256'), str) or len(record['sha256']) != 64
                or any(char not in '0123456789abcdef' for char in record['sha256'])):
            raise ValueError('Invalid size or SHA record: ' + relative)
        if relative == 'classifier.safetensors':
            continue
        if (not source.is_file() or source.stat().st_size != record['bytes']
                or sha256_file(source) != record['sha256']):
            raise ValueError('Metadata integrity verification failed: ' + relative)
        sources[relative] = source
    if manifest.get('checkpoint_sha256') != records['classifier.safetensors']['sha256']:
        raise ValueError('Manifest checkpoint SHA differs from its weight-file record')
    return manifest, manifest_bytes, sources


def reject_unlisted_files(root, manifest):
    """Do not publish stale pyc/temp files left by a previous partial transfer."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Bundle destination must not be a symlink')
    if not root.exists():
        return
    allowed = set(manifest['files']) | {'BUNDLE_MANIFEST.json'}
    for path in root.rglob('*'):
        relative = path.relative_to(root).as_posix()
        safe_bundle_path(root, relative)
        if not path.is_dir() and (not path.is_file() or relative not in allowed):
            raise ValueError('Destination contains an unlisted artifact; preserved without publishing: ' + relative)


def copy_listed_metadata(destination, manifest, manifest_bytes, sources):
    """Copy only the manifest and its explicitly listed non-weight files."""
    destination = Path(destination)
    reject_unlisted_files(destination, manifest)
    # Validate destination paths before making directories or replacing files.
    targets = {relative: safe_bundle_path(destination, relative) for relative in manifest['files']}
    manifest_target = safe_bundle_path(destination, 'BUNDLE_MANIFEST.json')
    destination.mkdir(parents=True, exist_ok=True)
    for relative, source in sources.items():
        target = targets[relative]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        record = manifest['files'][relative]
        if target.stat().st_size != record['bytes'] or sha256_file(target) != record['sha256']:
            raise ValueError('Copied metadata failed integrity verification: ' + relative)
    manifest_target.write_bytes(manifest_bytes)


def require_same_bundle(existing, expected):
    """A weight SHA alone cannot identify a release's code and inference rules."""
    identity_keys = ('format', 'candidate', 'variant', 'checkpoint_sha256',
                     'inference_engine', 'text_engine_contract', 'serialization',
                     'runtime_versions', 'runtime_image', 'runtime_requirements', 'loading_dtype',
                     'max_input_tokens', 'generated_tokens', 'production_approval',
                     'standalone_model_weights', 'reads_base_model_weights',
                     'reads_initial_h24_checkpoint', 'third_risk_class_validated',
                     'category_validated', 'safe_prefix_release_validated')
    different = [key for key in identity_keys if existing.get(key) != expected.get(key)]
    if existing.get('files') != expected.get('files'):
        different.append('files')
    if different:
        raise FileExistsError('Destination is a different bundle; refusing to overwrite or acknowledge it: '
                              + ', '.join(different))


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--pod', required=True)
    parser.add_argument('--metadata', type=Path, default=root / 'validation_results/output/round4_validation/bundle')
    parser.add_argument('--output', type=Path, default=root / 'release')
    parser.add_argument('--remote-bundle', default='/work/output/round4_validation/bundle',
                        help='Validated bundle directory on the project PVC; acknowledgement is written beside it')
    args = parser.parse_args()
    remote_bundle = PurePosixPath(args.remote_bundle)
    if (not remote_bundle.is_absolute() or '..' in remote_bundle.parts
            or not remote_bundle.is_relative_to('/work/output') or remote_bundle.name != 'bundle'):
        parser.error('--remote-bundle must be an absolute project output bundle directory')
    acknowledgement = str(remote_bundle.parent / 'weights_downloaded')
    bundle_manifest, manifest_bytes, metadata_sources = inspect_metadata(args.metadata)
    weight = bundle_manifest['files']['classifier.safetensors']
    expected = weight['sha256']
    k = ['kubectl', '--kubeconfig', str(Path.home() / '.kube/cls-og1rjus2-private-latest'), '-n', 'default']
    already_verified = False
    if args.output.exists():
        _, manifest, _ = verify_bundle(args.output)
        require_same_bundle(manifest, bundle_manifest)
        reject_unlisted_files(args.output, bundle_manifest)
        already_verified = True
    pod = json.loads(subprocess.check_output(k + ['get', 'pod', args.pod, '-o', 'json'], timeout=30))
    if pod['status']['phase'] != 'Running':
        if already_verified and pod['status']['phase'] == 'Succeeded':
            print(json.dumps({'already_verified': True, 'bundle': str(args.output), 'artifact_pod_completed': True}))
            return
        raise RuntimeError('The artifact transfer Pod is not running')
    for container in pod['spec']['containers']:
        if container.get('resources', {}).get('requests', {}).get('nvidia.com/gpu'):
            raise RuntimeError('Use a CPU-only transfer Pod; model downloading must not reserve a GPU')
    if already_verified:
        subprocess.run(k + ['exec', args.pod, '--', 'touch',
                            acknowledgement], check=True, timeout=30)
        print(json.dumps({'already_verified': True, 'bundle': str(args.output), 'artifact_pod_acknowledged': True}))
        return
    remote = str(remote_bundle / 'classifier.safetensors')
    remote_sha = subprocess.check_output(k + ['exec', args.pod, '--', 'sha256sum', remote], timeout=120).decode().split()[0]
    if remote_sha != expected:
        raise RuntimeError('Remote weight no longer matches the validated bundle metadata')
    partial = args.output.with_name('.' + args.output.name + '.partial-' + expected[:12])
    copy_listed_metadata(partial, bundle_manifest, manifest_bytes, metadata_sources)
    destination = partial / 'classifier.safetensors'
    offset = destination.stat().st_size if destination.exists() else 0
    if offset > weight['bytes']:
        raise RuntimeError('Partial weight is larger than the expected artifact')
    if offset < weight['bytes']:
        # tail opens the named project artifact directly; no shell interpolation.
        command = k + ['exec', args.pod, '--', 'tail', '-c', '+' + str(offset + 1), remote]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        progress = offset
        reported = offset
        try:
            with destination.open('ab') as output:
                while chunk := process.stdout.read(4 * 1024 * 1024):
                    output.write(chunk)
                    progress += len(chunk)
                    if progress - reported >= 128 * 1024 * 1024:
                        print(json.dumps({'downloaded_bytes': progress, 'expected_bytes': weight['bytes']}), flush=True)
                        reported = progress
            error = process.stderr.read().decode()
            if process.wait(timeout=30):
                raise RuntimeError('Artifact transfer failed; partial retained: ' + error)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    if destination.stat().st_size != weight['bytes']:
        raise RuntimeError('Incomplete artifact transfer; partial retained for resume')
    # Includes full SHA verification of weights, tokenizer, config, code and lock.
    reject_unlisted_files(partial, bundle_manifest)
    verify_bundle(partial)
    if args.output.exists():
        raise FileExistsError('Destination appeared while downloading; verified partial retained')
    partial.rename(args.output)
    subprocess.run(k + ['exec', args.pod, '--', 'touch',
                        acknowledgement], check=True, timeout=30)
    print(json.dumps({'bundle': str(args.output), 'checkpoint_sha256': expected,
                      'all_bundle_hashes_verified': True, 'model_calls': 0}), flush=True)


if __name__ == '__main__':
    main()
