import copy,json,random,time,unittest
from pathlib import Path
import numpy as np
from robustness_s4_extended.common import ROOT,CONFIG,load,atomic
from robustness_s4_extended.transforms import cutout,occlusion_fractions,sample_seed,validate
from robustness_s4_extended.op_prototype import orthonormal_basis,remove_projection,bounded_reconstruction
from robustness_s4_extended.metrics import paired

CFG=dict(n_holes=1,area_fraction=[.25,.25],probability=1.,fill_value=0,max_gt_occlusion=.5,max_attempts=8)

class CutoutTests(unittest.TestCase):
    def run_case(self,boxes,classes,hole,applied):
        img=np.full((64,80,3),160,np.uint8);b=np.array(boxes,dtype=float).reshape(-1,4);old=b.copy();before=img.copy()
        r,info=cutout(img,b,classes,CFG,np.random.default_rng(0),rectangles=[hole])
        self.assertEqual(info['applied'],applied);np.testing.assert_array_equal(img,before);np.testing.assert_array_equal(b,old)
        validate(r,b,classes)
        return r,info
    def test_single_person(self):self.run_case([[20,10,60,60]],[14],(0,0,20,20),True)
    def test_person_dog(self):self.run_case([[0,0,20,20],[30,30,60,60]],[14,11],(30,30,60,60),False)
    def test_person_horse_overlap(self):self.run_case([[15,10,60,60],[5,30,70,60]],[14,12],(25,30,45,50),True)
    def test_no_person(self):self.run_case([[20,10,60,60]],[6],(0,0,20,20),True)
    def test_small_target(self):self.run_case([[10,10,12,12]],[14],(10,10,12,12),False)
    def test_full_occlusion(self):self.run_case([[0,0,80,64]],[14],(0,0,80,64),False)
    def test_boundary(self):self.run_case([[5,5,50,50]],[14],(0,0,6,6),True)
    def test_empty(self):self.run_case([],[],(0,0,30,30),True)
    def test_all_classes_same_policy(self):
        for cls in range(20):self.run_case([[1,1,4,4]],[cls],(0,0,5,5),False)
    def test_union_not_double_counted(self):self.assertAlmostEqual(occlusion_fractions([[0,0,10,10]],[(0,0,6,10),(4,0,10,10)])[0],1.)
    def test_probability0(self):
        img=np.full((64,64,3),123,np.uint8);out,info=cutout(img,[],[],dict(CFG,probability=0),np.random.default_rng(1));np.testing.assert_array_equal(img,out);self.assertFalse(info['applied'])
    def test_seed_and_distribution(self):
        self.assertEqual(sample_seed(0,1,'id','cutout'),sample_seed(0,1,'id','cutout'));self.assertNotEqual(sample_seed(0,1,'id','cutout'),sample_seed(0,2,'id','cutout'))
        img=np.full((64,80,3),123,np.uint8)
        a,ai=cutout(img,[],[],CFG,np.random.default_rng(2));b,bi=cutout(img,[],[],CFG,np.random.default_rng(2));np.testing.assert_array_equal(a,b);self.assertEqual(ai,bi)
    def test_bad_class_and_geometry(self):
        for cls,box in [(20,[0,0,2,2]),(14,[3,3,2,2]),(1.2,[0,0,2,2])]:
            with self.assertRaises(ValueError):validate(np.zeros((10,10,3),np.uint8),[box],[cls])

class OPTests(unittest.TestCase):
    def test_rank_deficiency(self):
        q=orthonormal_basis([[1,0,0],[2,0,0],[0,0,0]]);self.assertEqual(q.shape,(3,1));np.testing.assert_allclose(q.T@q,np.eye(1))
    def test_idempotence_and_orthogonality(self):
        q=orthonormal_basis([[1,2,3],[0,1,0]]);x=np.arange(12).reshape(4,3);p=remove_projection(x,q);np.testing.assert_allclose(remove_projection(p,q),p,atol=1e-12);np.testing.assert_allclose(p@q,0,atol=1e-12)
    def test_zero_basis_and_bounded_reconstruction(self):
        x=np.array([[0.,1.,.5]]);q=orthonormal_basis(np.zeros((2,3)));np.testing.assert_array_equal(remove_projection(x,q),x);r,s=bounded_reconstruction(x,q,(1,1,3));self.assertEqual(s['effective_rank'],0);self.assertTrue(np.isfinite(r).all())
    def test_reject_nonorthogonal(self):
        with self.assertRaises(ValueError):remove_projection([[1,2]],[[1,1],[0,0]])

class MetricTests(unittest.TestCase):
    def record(self,v):return {key+'_per_class':[v]*20 for key in ['AP50','AP75','AP50_95','Recall','Precision']}
    def test_signed_no_clipping(self):
        r=paired(self.record(.5),self.record(.75));self.assertEqual(r['P'],-.5);self.assertEqual(r['U'],1.5);self.assertEqual(r['S'],-1.5);self.assertEqual(r['WCD'],-.5)
    def test_zero_denominator(self):self.assertIsNone(paired(self.record(0.),self.record(.5))['S'])
    def test_malformed_unit(self):
        with self.assertRaises(ValueError):paired(self.record(80.),self.record(.5))
    def test_identity(self):
        r=paired(self.record(.5),self.record(.5));self.assertEqual((r['P'],r['U'],r['S']),(0.,1.,0.))

class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ultralytics.utils.instance import Instances
        from robustness_s4_extended.integration import DetectionCutMix,SeededOperation
        cls.Instances=Instances;cls.DetectionCutMix=DetectionCutMix;cls.SeededOperation=SeededOperation
    def labels(self,boxes,classes,value=10):
        return dict(img=np.full((64,64,3),value,np.uint8),instances=self.Instances(np.array(boxes,dtype=np.float32).reshape(-1,4),segments=np.zeros((0,0,2),np.float32),bbox_format='xyxy',normalized=False),cls=np.array(classes,dtype=np.float32).reshape(-1,1),im_file='fixed_id.jpg')
    def dataset(self,labels):
        class Donor:
            def __len__(self):return 1
            def get_image_and_label(self,i):return copy.deepcopy(labels)
        return Donor()
    def test_cutmix_excludes_truncated_donor(self):
        a=self.labels([[30,30,60,60]],[14]);b=self.labels([[2,2,8,8],[5,5,15,15]],[11,12],220)
        t=self.DetectionCutMix(self.dataset(b),p=1.,donor_min_visible=.5);t._rand_bbox=lambda w,h:(0,0,10,10)
        r=t(a);self.assertEqual(r['cls'].reshape(-1).tolist(),[14,11]);np.testing.assert_array_equal(r['img'][:10,:10],b['img'][:10,:10]);validate(r['img'],r['instances'].bboxes,r['cls'])
    def test_cutmix_does_not_cover_primary(self):
        a=self.labels([[1,1,8,8]],[14]);before=a['img'].copy();b=self.labels([[2,2,8,8]],[12],220)
        t=self.DetectionCutMix(self.dataset(b),p=1.);t._rand_bbox=lambda w,h:(0,0,10,10)
        r=t(a);np.testing.assert_array_equal(r['img'],before);self.assertEqual(len(r['instances']),1)
    def test_cutmix_empty_source_and_all_truncated_skip(self):
        for donor in [self.labels([],[],220),self.labels([[5,5,50,50]],[14],220)]:
            t=self.DetectionCutMix(self.dataset(donor),p=1.);t._rand_bbox=lambda w,h:(0,0,10,10)
            r=t(self.labels([],[]));self.assertEqual(len(r['instances']),0);self.assertEqual(int(r['img'].max()),10)
    def test_mixup_multi_and_empty(self):
        from ultralytics.data.augment import MixUp
        for boxes,classes in [([[1,1,3,3],[3,3,20,20]],[14,12]),([],[])]:
            a=self.labels(boxes,classes,0);b=self.labels([[1,1,2,2]],[11],255);t=MixUp(self.dataset(b),p=1.)
            random.seed(0);np.random.seed(0);r=t(a);self.assertEqual(len(r['instances']),len(classes)+1);self.assertEqual(r['cls'].reshape(-1).tolist(),classes+[11]);self.assertTrue(0<=r['img'].min()<=r['img'].max()<=255);validate(r['img'],r['instances'].bboxes,r['cls'])
    def test_seed_scope_restores_rng(self):
        class D:pass
        d=D();d.ext_epoch=D();d.ext_epoch.value=0;d.ext_audit=ROOT/'execution'/'cpu_rng_audit'
        t=self.SeededOperation(lambda x:x,'mixup',d);t.calls=4
        random.seed(10);np.random.seed(10);a=random.getstate();b=np.random.get_state();t(self.labels([],[]));self.assertEqual(random.getstate(),a);np.testing.assert_array_equal(np.random.get_state()[1],b[1])
    def test_real_detection_dataset_pipeline_and_close_schedule(self):
        from PIL import Image
        from ultralytics.cfg import get_cfg
        from ultralytics.data import build
        from robustness_s4_extended.integration import configure,ExtendedDataset,SeededOperation
        from torch.utils.data import DataLoader
        fixture=ROOT/'execution'/'cpu_fixtures'/str(time.time_ns())
        for kind in ['images','labels']:(fixture/kind/'train').mkdir(parents=True)
        for i in range(4):
            Image.fromarray(np.full((64,64,3),100+i*20,np.uint8)).save(fixture/'images/train'/f'{i}.png')
            (fixture/'labels/train'/f'{i}.txt').write_text('14 .5 .5 .5 .5\n12 .3 .3 .2 .2\n' if i!=3 else '')
        q=load(CONFIG)
        meta={'names':{i:str(i) for i in range(20)},'path':str(fixture),'train':'images/train','val':'images/train'}
        for name in ['cutout','cutmix','mixup']:
            cfg=get_cfg(overrides=dict(imgsz=64,batch=2,workers=0,cache=False,epochs=200,close_mosaic=100,mixup=1. if name=='mixup' else 0.,cutmix=1. if name=='cutmix' else 0.));cfg.task='detect'
            configure(q['defenses'][name],fixture/'audit')
            original=build.YOLODataset
            try:
                build.YOLODataset=ExtendedDataset;ds=build.build_yolo_dataset(cfg,str(fixture/'images/train'),2,meta,mode='train',stride=32)
            finally:build.YOLODataset=original
            random.seed(0);np.random.seed(0)
            batch=next(iter(DataLoader(ds,batch_size=2,num_workers=0,collate_fn=ds.collate_fn)))
            self.assertEqual(batch['img'].shape[0],2);self.assertEqual(len(batch['cls']),len(batch['bboxes']));self.assertTrue((batch['cls']>=0).all() and (batch['cls']<20).all())
            ds.close_mosaic(copy.copy(cfg))
            wrappers=[t for t in ds.transforms.transforms if isinstance(t,SeededOperation)]
            self.assertEqual(len(wrappers),1)
            if name!='cutout':self.assertEqual(wrappers[0].operation.p,0.)
            # Actual candidate disabled path vs actual native baseline; no model training.
            zero=copy.deepcopy(q['defenses'][name]);zero['probability']=0.;configure(zero,fixture/'audit')
            hyp=get_cfg(overrides=dict(imgsz=64,batch=2,workers=0,cache=False,mixup=0.,cutmix=0.));hyp.task='detect'
            ref=build.build_yolo_dataset(hyp,str(fixture/'images/train'),2,meta,mode='train',stride=32)
            try:
                build.YOLODataset=ExtendedDataset;cand=build.build_yolo_dataset(hyp,str(fixture/'images/train'),2,meta,mode='train',stride=32)
            finally:build.YOLODataset=original
            random.seed(5);np.random.seed(5);a=ref[0]
            random.seed(5);np.random.seed(5);b=cand[0]
            import torch
            for key in ['img','cls','bboxes','batch_idx']:self.assertTrue(torch.equal(a[key],b[key]),(name,key))

class QueueContracts(unittest.TestCase):
    def test_frozen_config_rejects_drift(self):
        from robustness_s4_extended.common import validate_config
        c=load(CONFIG);validate_config(c)
        for key,value in [('batch',18),('workers',2),('amp',False)]:
            bad=copy.deepcopy(c);bad['baseline']['victim_protocol']['train_args'][key]=value
            with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(c);bad['OP_PROTOCOL_APPROVED']=True
        with self.assertRaises(ValueError):validate_config(bad)
    def test_op_not_callable_and_protocol_closed(self):
        c=load(CONFIG);self.assertFalse(c['OP_PROTOCOL_APPROVED']);self.assertFalse(c['defenses']['op']['enabled']);self.assertEqual(c['order'],['cutout','cutmix','mixup','op'])
        source=(ROOT/'robustness_s4_extended/worker.py').read_text();self.assertNotIn('op_prototype',source)
    def test_predecessor_identity_and_wait(self):
        from robustness_s4_extended.successor import Successor,Blocked
        c=copy.deepcopy(load(CONFIG));temp=ROOT/'execution'/'queue_fixtures'/str(time.time_ns());c['root']=str(temp);c['output_root']=str(temp/'outputs'/'q');c['predecessor']['output']=str(temp/'full');path=temp/'config.json';atomic(path,c);s=Successor(path)
        self.assertFalse(s.verify_wait_gate())
        p=c['predecessor'];state=dict(state='running',run_id=p['run_id'],source_commit=p['source_commit'],config_sha256=p['config_sha256']);atomic(temp/'full/pipeline_status.json',state);self.assertFalse(s.verify_wait_gate())
        state['state']='complete';atomic(temp/'full/pipeline_status.json',state);self.assertTrue(s.verify_wait_gate())
        # Native complete only allows acceptance checking, not training.
        self.assertFalse((s.root/'PREDECESSOR_ACCEPTANCE.json').exists())
        state['run_id']='wrong';atomic(temp/'full/pipeline_status.json',state)
        with self.assertRaises(Blocked):s.verify_wait_gate()
    def test_no_cached_clean_control(self):
        c=load(CONFIG);self.assertEqual(c['baseline']['victim_protocol']['train_args']['batch'],36);self.assertEqual(c['smoke_batches_per_arm'],20)
        source=(ROOT/'robustness_s4_extended/successor.py').read_text();self.assertIn("for arm in ['clean','S4']",source)

if __name__=='__main__':unittest.main()
