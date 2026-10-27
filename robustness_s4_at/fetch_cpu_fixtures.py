"""Download two hash-pinned public VOC train fixtures through existing RRCTL transport."""
import argparse
from pathlib import Path
from .common import load, sha, write_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from remote_run_control.profiles import ProfileStore
    from remote_run_control.transport import transport_for
    transport = transport_for(ProfileStore(args.profiles).load('school-s1'))
    records = load(args.manifest)['records'][:2]
    root = '/data/guoqinglong/datasets/VOC_0712_Kaggle_Ready/'
    evidence = []
    for record in records:
        row = dict(image_id=record['image_id'])
        for key, subdir in (('image', 'images/train'), ('label', 'labels/train')):
            remote = record[key + '_path']
            if not remote.startswith(root + subdir + '/') or Path(remote).stem != record['image_id']:
                raise ValueError('Fixture escaped the accepted public VOC train split')
            target = args.out / subdir / Path(remote).name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(transport.download(remote))
            if sha(target) != record[key + '_sha256']:
                raise ValueError('Public VOC fixture byte identity mismatch: ' + remote)
            row[key + '_path'] = str(target.resolve())
            row[key + '_sha256'] = sha(target)
        evidence.append(row)
    target = args.out / 'fixture_identity.json'
    result = dict(kind='hash-verified public VOC train fixtures, no validation images',
                  source_manifest_sha256=sha(args.manifest), records=evidence,
                  model_forwards=0, GPU_initialized=False)
    if target.exists():
        if load(target) != result:
            raise ValueError('Fixture identity collision')
    else:
        write_once(target, result)
    print(result)


if __name__ == '__main__':
    main()
