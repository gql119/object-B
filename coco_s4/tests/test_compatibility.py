import copy,itertools,json,math,types,unittest
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from coco_s4.cache import OnDemandSafeBGTripletCache
from coco_s4.data import COCO_IDS,category_mapping,audit_metadata,audit_pixels,annotation_to_yolo
from coco_s4.evaluation import paired_metrics,per_class
from coco_s4.protocol import expected_noise_schedule,admit_gpu
from ue_framework.methods.safebg import SafeBGTripletCache,safe_background_direction,select_strict_tsfl
from ue_framework.methods.legacy_kproto_ret import _StrictTargetSupport
from ue_framework.methods.s4_pixel_audit import domains,png_project
from ue_framework.methods.tfss_spatial import spatial_weights

def categories():return [{'id':c,'name':'person' if i==0 else f'class_{i}'} for i,c in enumerate(COCO_IDS)]
def example(split,offset=0):
    im={'id':1+offset,'file_name':f'{1+offset:012d}.jpg','width':160,'height':96,'coco_url':f'http://images.cocodataset.org/{split}/x.jpg'}
    anns=[{'id':100+offset+i,'image_id':im['id'],'category_id':c,'bbox':[10,10,20,30],'area':600,'iscrowd':0} for i,c in enumerate(COCO_IDS)]
    return {'images':[im],'annotations':anns,'categories':categories()}

class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(52)
        cls.bases={c:torch.linalg.qr(torch.randn(24,4,dtype=torch.float64)).Q for c in range(20) if c!=14}
        cls.old=SafeBGTripletCache(cls.bases);cls.new=OnDemandSafeBGTripletCache(cls.bases,max_entries=4)
    def test_old_19_cache_exact_ordered_variants(self):
        for order in itertools.permutations((0,1,19)):
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(self.old.lookup(order),self.new.lookup(order))))
    def test_complete_direction_and_diagnostics_exact(self):
        torch.manual_seed(4);x=torch.randn(4,24,dtype=torch.float64);b=x+torch.randn_like(x)
        a,ad=safe_background_direction(x,b,self.bases,mode='low_rank',safe_rank=8,removal_alpha=.5,triplet_cache=self.old)
        z,zd=safe_background_direction(x,b,self.bases,mode='low_rank',safe_rank=8,removal_alpha=.5,triplet_cache=self.new)
        self.assertTrue(torch.equal(a,z));self.assertEqual(ad['active_classes'],zd['active_classes'])
        for k in ad:
            if torch.is_tensor(ad[k]):self.assertTrue(torch.equal(ad[k],zd[k]),k)
    def test_79_classes_bounded_and_call_order_independent(self):
        bases={i+1:self.bases[list(self.bases)[i%19]] for i in range(79)}
        a=OnDemandSafeBGTripletCache(bases,max_entries=1);b=OnDemandSafeBGTripletCache(dict(reversed(list(bases.items()))),max_entries=2)
        first=a.lookup([79,1,2]);a.lookup([3,4,5]);second=a.lookup([79,1,2]);b.lookup([3,4,5]);other=b.lookup([79,1,2])
        self.assertTrue(all(torch.equal(x,y) and torch.equal(x,z) for x,y,z in zip(first,second,other)))
        self.assertEqual(len(a.entries),1);self.assertEqual(a.sha256,b.sha256)
    def test_missing_and_zero_rank_rejected(self):
        bad=dict(self.bases);bad[1]=torch.zeros(24,4,dtype=torch.float64)
        with self.assertRaises(ValueError):OnDemandSafeBGTripletCache(bad)
        with self.assertRaises(ValueError):self.new.lookup([0,0,1])

class DataTests(unittest.TestCase):
    def test_category_id_gaps_person0_all80(self):
        m=category_mapping(example('train2017'));self.assertEqual(m[0]['name'],'person');self.assertEqual(m[-1]['category_id'],90);self.assertEqual(m[-1]['yolo_class_id'],79)
    def test_complete_metadata_and_crowd_ignore(self):
        t=example('train2017');v=example('val2017',1000)
        t['annotations'].append(dict(t['annotations'][0],id=9999,iscrowd=1))
        t['annotations'].append(dict(t['annotations'][0],id=9998,ignore=1))
        a,labels=audit_metadata(t,v,train_count=1,val_count=1)
        self.assertEqual(len(labels['train'][1]),80);self.assertEqual(a['splits']['train']['crowd_annotations'],1);self.assertTrue(a['disjoint'])
    def test_no_person_and_empty_image_retained(self):
        t=example('train2017');v=example('val2017',1000)
        for i in [2,3]:t['images'].append(dict(t['images'][0],id=i,file_name=f'{i:012d}.jpg'))
        t['annotations'].append(dict(t['annotations'][1],id=9999,image_id=2))
        a,labels=audit_metadata(t,v,train_count=3,val_count=1)
        self.assertEqual(len(labels['train']),3);self.assertEqual(labels['train'][3],[]);self.assertEqual(a['splits']['train']['no_person_images'],2)
    def test_duplicate_foreign_and_overlap_rejected(self):
        for mutate in [lambda t:t['images'].append(t['images'][0]),lambda t:t['annotations'].append(t['annotations'][0]),lambda t:t['annotations'][0].update(category_id=12)]:
            t=example('train2017');mutate(t)
            with self.assertRaises(ValueError):audit_metadata(t,example('val2017',1000),train_count=1,val_count=1)
        t=example('train2017');v=example('val2017')
        with self.assertRaises(ValueError):audit_metadata(t,v,train_count=1,val_count=1)
    def test_bbox_values_no_silent_clipping(self):
        im=example('train2017')['images'][0];a=example('train2017')['annotations'][0]
        self.assertEqual(annotation_to_yolo(a,im,{1:0}), (0,20/160,25/96,20/160,30/96))
        for box in [[-1,0,2,2],[0,0,0,2],[159,0,2,2],[0,0,float('nan'),2]]:
            with self.assertRaises(ValueError):annotation_to_yolo(dict(a,bbox=box),im,{1:0})

class SpatialTests(unittest.TestCase):
    def test_current_original_S4_math_sources_exact(self):
        import hashlib
        root=Path(__file__).resolve().parents[2]
        lock=json.loads((root/'docs/S4_COCO_MINITRAIN_20261023/ORIGINAL_S4_METHOD_SOURCE_LOCK.json').read_text())
        for rel,expected in lock['files'].items():
            self.assertEqual(hashlib.sha256((root/rel).read_bytes()).hexdigest(),expected,rel)
    def test_real_parent_support_all_geometry_cases(self):
        for h,w in [(10,30),(31,9),(1,5),(64,64)]:
            support=types.SimpleNamespace(target_class_id=0)
            for anns in [[],[{'cls':79,'bbox':[.5,.5,.8,.8]}],[{'cls':0,'bbox':[.5,.5,.8,.8]}],[{'cls':0,'bbox':[.3,.5,.5,.8]},{'cls':79,'bbox':[.6,.5,.5,.8]},{'cls':70,'bbox':[.6,.5,.5,.8]}],[{'cls':0,'bbox':[.01,.01,.02,.02]}]]:
                s=_StrictTargetSupport._bbox_half_support(support,(h,w,3),anns);t,n=domains((h,w,3),anns,0,80)
                self.assertTrue(np.array_equal(s,t.astype(np.float32)*np.where(n,.5,1)))
                clean=np.full((h,w,3),128,dtype=np.uint8);p=png_project(clean,np.ones((h,w,3)),s*8/255)
                self.assertEqual(audit_pixels(clean,p,s)['outside_nonzero'],0)
    def test_VOC_default_domains_unchanged(self):
        a=[{'cls':14,'bbox':[.5,.5,.8,.8]},{'cls':19,'bbox':[.7,.5,.4,.8]}]
        x=domains((10,20,3),a);y=domains((10,20,3),a,14,20)
        self.assertTrue(all(np.array_equal(p,q) for p,q in zip(x,y)))
    def test_Full_budget_and_quarter_overlap_rejected(self):
        c=np.zeros((2,2,3),dtype=np.uint8);s=np.array([[1,.5],[0,1]],dtype=np.float32);p=c.copy();p[0,1]=8
        with self.assertRaises(ValueError):audit_pixels(c,p,s)
        with self.assertRaises(ValueError):audit_pixels(c,c,np.full((2,2),.25))
    def test_TFSS_formula_gradient_and_bounds(self):
        q=torch.rand(3,8);r=torch.rand(3,8);s=torch.ones(3,8)
        w=spatial_weights(q,r,s,mode='TFSS-U',eta=.85)
        self.assertTrue(torch.allclose(w,1-.15*r*(1-q),atol=1e-7,rtol=0));self.assertGreaterEqual(float(w.min()),.85)
        x=torch.randn(3,8,requires_grad=True);(x*w).sum().backward();self.assertTrue(torch.equal(x.grad,w))

class EvaluationAndGates(unittest.TestCase):
    def test_original_victim_component_exact(self):
        import hashlib
        root=Path(__file__).resolve().parents[2]
        lock=json.loads((root/'docs/S4_COCO_MINITRAIN_20261023/ORIGINAL_VICTIM_COMPONENT.json').read_text())
        for rel,expected in lock['files'].items():
            self.assertEqual(hashlib.sha256((root/rel).read_bytes()).hexdigest(),expected,rel)
    def test_saved_method_hyperparameters_and_architecture_match_parent(self):
        import yaml
        root=Path(__file__).resolve().parents[2]
        old=yaml.safe_load((root/'ue_framework/configs/exp_voc_person_b2_to_safe_soft05_eps8_seed0_20261004.yaml').read_text())
        new=yaml.safe_load((root/'docs/S4_COCO_MINITRAIN_20261023/S4_COCO_EXPERIMENT_CONFIG.yaml').read_text())
        a=copy.deepcopy(old['methods']['legacy_kproto_ret']);b=copy.deepcopy(new['methods']['legacy_kproto_ret'])
        for x in (a,b):
            x['semantic'].pop('artifact_dir');x['semantic']['strict_tsfl'].pop('layer_selection_path');x['semantic']['safe_bg'].pop('reference_artifact_dir')
        self.assertEqual(a,b);self.assertEqual(b['support_mode'],'bbox_half_overlap')
        h=yaml.safe_load((root/'configs/voc_yolov8n_20cls.yaml').read_text());h['nc']=80
        self.assertEqual(h,yaml.safe_load((root/'configs/coco_yolov8n_80cls.yaml').read_text()))
    def test_signed_metrics_and_zero_denominators(self):
        a=[{'yolo_class_id':i,'AP50':.5,'AP75':.3,'AP50_95':.2} for i in range(80)];b=copy.deepcopy(a);b[0]['AP50']=.1
        m=paired_metrics(a,b);self.assertEqual(m['P'],.8);self.assertEqual(m['U'],1);self.assertAlmostEqual(m['S'],8/9)
        b[0]['AP50']=.8;self.assertLess(paired_metrics(a,b)['P'],0)
        a[0]['AP50']=0;self.assertIsNone(paired_metrics(a,b)['P'])
    def test_COCO_array_reductions_and_maxDets(self):
        p=types.SimpleNamespace(catIds=list(COCO_IDS),maxDets=[1,10,100],areaRngLbl=['all','small','medium','large'],iouThrs=np.linspace(.5,.95,10))
        e=types.SimpleNamespace(params=p,eval={'precision':np.ones((10,101,80,4,3))*.4,'recall':np.ones((10,80,4,3))*.6})
        x=per_class(e,category_mapping(example('train2017')));self.assertEqual(len(x),80);self.assertAlmostEqual(x[79]['AP50_95'],.4)
        p.maxDets[-1]=300
        with self.assertRaises(ValueError):per_class(e,category_mapping(example('train2017')))
    def test_epoch_schedule_is_data_dependent_only_on_COCO(self):
        self.assertEqual(expected_noise_schedule(20,6095)['optimizer_steps'],9144)
        self.assertEqual(expected_noise_schedule(80,17)['optimizer_steps'],48)
    def test_idle_or_finished_file_not_sufficient(self):
        q={'predecessor_run_id':'bound'};s={'pipeline_run_id':'bound','active_job':None,'stages':{c:'ACCEPTED' for c in ['cutout','cutmix','mixup']},'state':'BLOCKED_PROTOCOL_APPROVAL'}
        self.assertEqual(admit_gpu(s,q,{}),'WAITING_PREVIOUS_ARCHIVE')
        self.assertEqual(admit_gpu(s,q,{'first3_archive_verified':True}),'BLOCKED_NEW_EXPERIMENT_GATES')
        s['active_job']='mixup_poison';self.assertEqual(admit_gpu(s,q,{}),'WAITING_PREVIOUS_PIPELINE')

if __name__=='__main__':unittest.main()
