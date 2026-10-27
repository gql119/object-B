"""Hash-check retained S4 metadata and fetch two immutable poison CPU fixtures."""
import argparse
from pathlib import Path
from .common import protocol, load, sha, write_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--clean-fixtures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    from remote_run_control.profiles import ProfileStore
    from remote_run_control.transport import transport_for
    transport = transport_for(ProfileStore(a.profiles).load('school-s1'))
    p = protocol()
    for rel in ('materialization_manifest.json', 'dataset_integrity.json'):
        path = a.out / 'metadata' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            data = transport.download(p['parent_output'] + '/' + rel)
            if len(data) > 16 * 2**20:
                raise ValueError('Unexpectedly large metadata report; no dataset bulk transfer')
            path.write_bytes(data)
        if sha(path) != p['reused_artifacts'][rel]:
            raise ValueError('Current server S4 metadata hash drift: ' + rel)
    manifest = load(a.out / 'metadata/materialization_manifest.json')
    if len(manifest) != 16551 or len({r['id'] for r in manifest}) != 16551:
        raise ValueError('Original S4 materialization record count mismatch')
    rows = {r['id']: r for r in manifest}
    records = []
    for clean in load(a.clean_fixtures / 'fixture_identity.json')['records']:
        row = rows[clean['image_id']]
        remote = Path(row['output'])
        if not remote.is_relative_to(Path(p['poison_root']) / 'images/train'):
            raise ValueError('Poison sample escaped accepted dataset')
        image = a.out / 'images/train' / remote.name
        label = a.out / 'labels/train' / (row['id'] + '.txt')
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        if not image.exists(): image.write_bytes(transport.download(remote.as_posix()))
        if not label.exists(): label.write_bytes(Path(clean['label_path']).read_bytes())
        if sha(image) != row['output_sha256'] or sha(label) != row['label_sha256']:
            raise ValueError('Current S4 CPU fixture byte identity drift')
        records.append(dict(image_id=row['id'], image_path=str(image.resolve()),
                            image_sha256=sha(image), label_path=str(label.resolve()), label_sha256=sha(label)))
    identity = dict(kind='accepted Original S4 static poison train fixtures, no validation', records=records,
                    source_manifest_sha256=p['reused_artifacts']['materialization_manifest.json'])
    target = a.out / 'fixture_identity.json'
    if target.exists():
        if load(target) != identity: raise ValueError('Poison fixture identity collision')
    else:
        write_once(target, identity)
    print(dict(current_server_metadata_SHA_exact=True, materialized_records=16551,
               original_poison_fixture_images=len(records), GPU_work=0, static_data_changed=False))


if __name__ == '__main__':
    main()
