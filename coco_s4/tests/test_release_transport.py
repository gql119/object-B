import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from coco_s4.phase_contract import OWNED
from coco_s4.release_transport import (REMOTE_INSPECT, REMOTE_OWNED, anchor_path,
                                       close_and_register, validate_anchor_outside_lifecycle)
from coco_s4.runtime import ROOT, sha


class FakeTransport:
    def __init__(self):
        self.files = {}
        self.uploads = []

    def run(self, argv):
        paths = argv[5:]
        result = {name: self.files.get(name) for name in paths}
        return SimpleNamespace(returncode=0, stdout=json.dumps(result).encode(), stderr=b'')

    def upload(self, local_path, remote_path, expected_sha256):
        digest = sha(local_path)
        if digest != expected_sha256:
            raise AssertionError('test upload SHA mismatch')
        self.files[remote_path] = digest
        self.uploads.append(remote_path)


class ReleaseTransportTests(unittest.TestCase):
    def test_anchor_is_sibling_to_all_lifecycle_roots(self):
        tag = 'coco_teacher_eval_testtag'
        remote = {
            'repo_root': (OWNED/'releases'/tag).as_posix(),
            'output_root': (OWNED/'validation'/tag).as_posix(),
            'stage_root': (OWNED/'stage'/tag).as_posix(),
            'control_root': (OWNED/'control'/tag).as_posix(),
        }
        anchor = anchor_path(tag)
        self.assertIn('/releases/_anchors/', anchor)
        self.assertEqual(validate_anchor_outside_lifecycle(anchor, remote), anchor)

    def test_anchor_inside_repo_lifecycle_is_rejected(self):
        tag = 'coco_teacher_eval_testtag'
        repo = (OWNED/'releases'/tag).as_posix()
        remote = {'repo_root': repo, 'output_root': (OWNED/'validation'/tag).as_posix(),
                  'stage_root': (OWNED/'stage'/tag).as_posix(),
                  'control_root': (OWNED/'control'/tag).as_posix()}
        with self.assertRaisesRegex(ValueError, 'overlaps a lifecycle root'):
            validate_anchor_outside_lifecycle(repo+'/source_release.json', remote)

    def test_registration_uploads_release_before_ready_and_is_idempotent(self):
        tag = 'coco_teacher_eval_testregister'
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp)/tag
            candidate.mkdir()
            release_path = candidate/'source_release.json'
            smoke_path = candidate/'smoke_summary.json'
            review_path = candidate/'review.json'
            config_path = 'coco_s4/configs/teacher_eval.json'
            release = {'phase':'teacher_eval','workflow_run_id':'s4_coco_experiment_a_seed0_20261025_v3',
                       'source_commit':'test-commit','phase_config_path':config_path,
                       'teacher_dependency':{'accepted_sha256':'a'*64,'checkpoint_sha256':'b'*64}}
            release_path.write_text(json.dumps(release), encoding='utf-8')
            smoke_path.write_text('{"state":"passed"}', encoding='utf-8')
            review = {'phase':'teacher_eval','source_commit':'test-commit','release_sha256':sha(release_path),
                      'phase_config_sha256':sha(ROOT/config_path),'smoke_sha256':sha(smoke_path),
                      'reviewer_count':1,'reviewer_independence':True,'pre_run_result':'pass',
                      'scientifically_correct':True,'allow_run':True,'scientific_verdict':'allow_run','blockers':[]}
            review_path.write_text(json.dumps(review), encoding='utf-8')
            spec = {'source':{'commit':'test-commit','repo_root':str(ROOT)},
                    'remote':{'repo_root':(OWNED/'releases'/tag).as_posix(),
                              'output_root':(OWNED/'validation'/tag).as_posix()}}
            remote_smoke = spec['remote']['output_root']+'/result/smoke_summary.json'
            transport = FakeTransport(); transport.files[remote_smoke] = sha(smoke_path)
            close_and_register(transport,tag=tag,phase='teacher_eval',module='coco_s4.phase_teacher_eval',
                spec=spec,release_path=release_path,smoke_path=smoke_path,review_path=review_path,
                previous_stage_receipt_sha256='a'*64)
            self.assertEqual(transport.uploads[0],spec['remote']['repo_root']+'/source_release.json')
            self.assertTrue(transport.uploads[-1].endswith('/teacher_eval_READY.json'))
            first_upload_count=len(transport.uploads)
            close_and_register(transport,tag=tag,phase='teacher_eval',module='coco_s4.phase_teacher_eval',
                spec=spec,release_path=release_path,smoke_path=smoke_path,review_path=review_path,
                previous_stage_receipt_sha256='a'*64)
            self.assertEqual(len(transport.uploads),first_upload_count)


if __name__ == '__main__':
    unittest.main()
