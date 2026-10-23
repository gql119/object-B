"""CPU scientific contracts; retained outputs, no official AP and no downloads."""
import copy
import hashlib
import json
import random
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from ue_framework.methods.legacy_kproto_ret import LegacyKProtoTrainer,LegacyKProtoGenerator
from ue_framework.methods.s4_tfss import S4Trainer,CleanMapCache
from ue_framework.methods.s4_tfss_full import (S4FullTrainer,S4FullGenerator,FullTargetSupport,
    full_support,export_project,spatial_diagnostics,original_training_eligible)
from ue_framework.methods.s4_pixel_audit import domains
from ue_framework.methods.tfss_spatial import spatial_weights
from ue_framework.s4_full_contract import load_full_config,verify_queue

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/"docs/S4_TFSS_FULL_20261021"
OWNED=ROOT/"execution/cpu_tests"
OWNED.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)


def retained():
    return Path(tempfile.mkdtemp(prefix="full_",dir=OWNED))


def bare(cls,mode,training=True):
    obj=cls.__new__(cls)
    for key,value in dict(support_mode=mode,target_class_id=14,eps=8/255,imgsz=32,
        device=torch.device("cpu"),jnd_floor=.4,jnd_ceiling=1.,lambda_freq=1.,
        freq_amp_buffer=1.,tanh_temp=1.,is_universal_training=training,
        strict_instance_mask=False,use_prebaked_instance_mask=False,instance_mask_dir="",
        ring_width=4).items():setattr(obj,key,value)
    obj.coords=[(1,2),(2,3)]
    obj.fourier_coeff=torch.tensor([[.3,-.2,.6],[-.1,.4,.8]],requires_grad=True)
    obj.suppress_small=torch.zeros(1,3,4,4)
    obj.tfss_mode="TFSS-U"
    return obj


P={"cls":14,"bbox":[.5,.5,.75,.75]}
DOG={"cls":11,"bbox":[.5,.5,.5,.5]}
HORSE={"cls":12,"bbox":[.4,.5,.6,.5]}


class FullContracts(unittest.TestCase):
    def test_config_only_one_variable_and_canonical45_unchanged(self):
        lock=json.loads((DOC/"S4_TFSS_FULL_LOCK.json").read_text())
        cfg=load_full_config(ROOT,lock)
        self.assertEqual(cfg["methods"]["legacy_kproto_ret"]["support_mode"],"bbox_full_target")
        for rel,expected in json.loads((ROOT/"docs/tfss/S0_CANONICAL_SOURCE_SHA256.json").read_text()).items():
            self.assertEqual(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),expected,rel)

    def test_support_all_geometries(self):
        scenarios=[[P],[P,DOG],[P,HORSE],[P,dict(P,bbox=[.4,.5,.5,.5])],
            [P,DOG,HORSE,{"cls":1,"bbox":[.5,.5,.3,.9]}],[],[DOG],
            [{"cls":14,"bbox":[0.,0.,.03,.03]}],
            [{"cls":14,"bbox":[1.,1.,.03,.03]},DOG]]
        for anns in scenarios:
            with self.subTest(annotations=anns):
                full=bare(S4FullTrainer,"bbox_full_target")
                half=bare(S4Trainer,"bbox_half_overlap")
                support,ring,source=full._build_support((32,39,3),anns)
                old=half._build_support((32,39,3),anns)[0]
                target,other=domains((32,39,3),anns)
                np.testing.assert_array_equal(support,target.astype(np.float32))
                np.testing.assert_array_equal(old>0,support>0)
                self.assertTrue(np.all(support[target&other]==1))
                self.assertFalse(ring.any())
                self.assertEqual(original_training_eligible((32,39,3),anns),old.sum()>10)

    def test_threshold_crossing_tiny_overlap_preserves_original_sample_policy(self):
        ann=[{"cls":14,"bbox":[.5,.5,.1,.08]},dict(DOG,bbox=[.5,.5,1.,1.])]
        full=full_support((32,39,3),ann)
        old=bare(S4Trainer,"bbox_half_overlap")._build_support((32,39,3),ann)[0]
        self.assertGreater(float(full.sum()),10)
        self.assertLessEqual(float(old.sum()),10)
        self.assertFalse(original_training_eligible((32,39,3),ann))

    def compose_pair(self,anns,training=True,edge_pixels=False):
        image=torch.linspace(.05,.95,32*39*3).reshape(1,3,32,39)
        if edge_pixels:image[0,:,0,:]=0;image[0,:,-1,:]=1
        outputs=[]
        for cls,mode in [(S4Trainer,"bbox_half_overlap"),(S4FullTrainer,"bbox_full_target")]:
            obj=bare(cls,mode,training)
            support_np=obj._build_support((32,39,3),anns)[0]
            support=torch.from_numpy(support_np)[None,None]
            q=torch.linspace(0,1,32*39).reshape(32,39);risk=1-q
            obj.tfss_weight=spatial_weights(q,risk,support[0,0],mode="TFSS-U")[None,None]
            random.seed(18);torch.manual_seed(18)
            result=obj._compose_delta_batched(image,support,support*0,obj.coords,obj.fourier_coeff,obj.suppress_small,17)
            grad=torch.autograd.grad(result[1].square().sum(),obj.fourier_coeff)[0]
            outputs.append((obj,result,grad))
        return image,outputs

    def test_nonoverlap_exact_forward_raw_jnd_and_gradient(self):
        anns=[P,{"cls":11,"bbox":[.98,.98,.02,.02]}]
        for training in (False,True):
            _,pairs=self.compose_pair(anns,training)
            for a,b in zip(pairs[0][1],pairs[1][1]):torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(pairs[0][2],pairs[1][2],rtol=0,atol=0)

    def test_overlap_only_doubles_base_retains_W_and_gradients(self):
        image,pairs=self.compose_pair([P,DOG,HORSE])
        target,other=domains((32,39,3),[P,DOG,HORSE]);mask=torch.from_numpy(target&other)
        old,new=pairs[0][1][1],pairs[1][1][1]
        torch.testing.assert_close(new[:,:,mask],old[:,:,mask]*2,rtol=0,atol=0)
        torch.testing.assert_close(new[:,:,~mask],old[:,:,~mask],rtol=0,atol=0)
        self.assertTrue(torch.equal(pairs[0][0].tfss_weight,pairs[1][0].tfss_weight))
        self.assertGreater(float(pairs[1][2].abs().sum()),0)
        self.assertIs(pairs[1][0]._last_poison,pairs[1][1][2])
        self.assertLessEqual(float(new.abs().max()),8/255+1e-8)
        self.assertEqual(float(new[:,:,~torch.from_numpy(target)].abs().sum()),0)

    def test_image_clamp_boundary_and_no_person(self):
        for anns in ([P,DOG],[],[DOG]):
            image,pairs=self.compose_pair(anns,edge_pixels=True)
            actual=pairs[1][1][2]-image
            target,_=domains((32,39,3),anns)
            self.assertEqual(float(actual[:,:,~torch.from_numpy(target)].abs().sum()),0)
            self.assertLessEqual(float(actual.abs().max()),8/255+1e-7)

    def test_same_generator_and_trainer_composition_and_export_PNG(self):
        anns=[P,DOG,HORSE]
        image=np.linspace(.1,.9,32*39*3,dtype=np.float32).reshape(32,39,3)
        trainer=bare(S4FullTrainer,"bbox_full_target",False)
        gen=bare(S4FullGenerator,"bbox_full_target",False)
        support=trainer._build_support(image.shape,anns)[0]
        q=torch.zeros_like(torch.from_numpy(support));risk=torch.ones_like(q)
        weight=spatial_weights(q,risk,torch.from_numpy(support),mode="TFSS-U")
        trainer.tfss_weight=weight[None,None]
        x=torch.from_numpy(image).permute(2,0,1)[None]
        trained=trainer._compose_delta_batched(x,torch.from_numpy(support)[None,None],torch.zeros(1,1,32,39),
            trainer.coords,trainer.fourier_coeff,trainer.suppress_small,0)
        result=gen.generate(image,anns,0,24,8/255,"mask")
        np.testing.assert_array_equal(result.perturbation*weight.numpy()[:,:,None],
                                     trained[1][0].detach().permute(1,2,0).numpy())
        clean=np.rint(image*255).astype(np.uint8)
        png=export_project(clean,np.clip(image+result.perturbation*weight.numpy()[:,:,None],0,1),anns,8/255)
        folder=retained();Image.fromarray(png).save(folder/"result.png")
        decoded=np.asarray(Image.open(folder/"result.png"))
        np.testing.assert_array_equal(png,decoded)
        target,_=domains(image.shape,anns)
        delta=decoded.astype(np.int16)-clean.astype(np.int16)
        self.assertEqual(int(np.abs(delta[~target]).max()),0)
        self.assertLessEqual(int(np.abs(delta).max()),8)

    def test_export_allows_overlap_above_four_without_exceeding_eight(self):
        clean=np.full((32,39,3),128,dtype=np.uint8);anns=[P,DOG]
        candidate=np.full_like(clean,255).astype(np.float32)/255
        result=export_project(clean,candidate,anns,8/255)
        target,other=domains(clean.shape,anns);d=result.astype(np.int16)-clean.astype(np.int16)
        self.assertEqual(int(d[target&other].max()),8)
        self.assertEqual(int(np.abs(d[~target]).max()),0)

    def test_tiny_Person_export_does_not_silently_skip(self):
        gen=bare(S4FullGenerator,"bbox_full_target",False)
        image=np.full((32,32,3),.5,dtype=np.float32)
        result=gen.generate(image,[{"cls":14,"bbox":[0,0,.03,.03]}],0,24,8/255,"mask")
        self.assertTrue(result.extras["is_poisoned"])
        self.assertTrue(np.any(result.perturbation!=0))
        empty=gen.generate(image,[DOG],0,24,8/255,"mask")
        self.assertFalse(empty.extras["is_poisoned"])
        np.testing.assert_array_equal(empty.poisoned_image,image)

    def test_real_GT_geometry_overlap_diagnostics_even_binary_support(self):
        anns=[P,DOG,HORSE];target,other=domains((32,39,3),anns)
        d=target[:,:,None]*np.ones((32,39,3))*.02
        row=spatial_diagnostics(np.full_like(d,.5),d,anns,
            q=np.ones(target.shape)*.3,risk=np.ones(target.shape)*.7,weight=np.ones(target.shape)*.9265)
        self.assertGreater(row["overlap_area_pixels"],0)
        self.assertAlmostEqual(row["overlap_RMS_delta"],.02)
        self.assertAlmostEqual(row["overlap_mean_W_U"],.9265)
        self.assertEqual(set(row["overlap_by_NT_class"]),{"11","12"})

    def test_generator_rejects_old_half_checkpoint(self):
        # Constructor's inherited mode guard is independently audited in source;
        # runtime metadata mismatch is exercised without external model weights.
        from ue_framework.config import load_config
        from ultralytics import YOLO
        cfg=load_config(str(ROOT/"ue_framework/configs/exp_voc_person_s4_tfss_full_seed0_20261021.yaml"))
        cfg["surrogate"]["imgsz"]=32
        folder=retained();checkpoint=folder/"half.pt"
        torch.save(dict(method="legacy_kproto_ret",support_mode="bbox_half_overlap",eps=cfg["experiment"]["eps"],
            feature_layer="model.21",coords=[(1,2)],fourier_coeff=torch.ones(1,3),suppress_small=torch.zeros(1,3,4,4)),checkpoint)
        model=YOLO(str(ROOT/"configs/voc_yolov8n_20cls.yaml")).model
        with self.assertRaisesRegex(ValueError,"support mode"):
            S4FullGenerator(cfg,cfg["methods"]["legacy_kproto_ret"],torch.device("cpu"),model,str(checkpoint))

    def test_queue_fail_closed_and_no_idle_GPU_bypass(self):
        folder=retained();state_path=folder/"pipeline.json";now=time.time()
        lock=json.loads((DOC/"S4_TFSS_FULL_LOCK.json").read_text());lock["queue"]["pipeline_state_path"]=str(state_path)
        state=dict(pipeline_run_id=lock["queue"]["pipeline_id"],state="RUNNING",active_job=None,stages={})
        state_path.write_text(json.dumps(state))
        receipt=dict(source_commit="candidate",config_sha256="config",pipeline_state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
            checked_at_unix=now,queue_conflicts=[],planned_queue_checked=True,own_active_training_roots=0,observed_via="RRCTL")
        with self.assertRaisesRegex(ValueError,"not fully accepted"):verify_queue(lock,receipt,source_commit="candidate",config_sha256="config",now=now)
        state.update(state="COMPLETED_AND_ACCEPTED",stages={stage:{"state":"ACCEPTED"} for stage in lock["queue"]["required_stages"]})
        state_path.write_text(json.dumps(state));receipt["pipeline_state_sha256"]=hashlib.sha256(state_path.read_bytes()).hexdigest()
        verify_queue(lock,receipt,source_commit="candidate",config_sha256="config",now=now)
        receipt["queue_conflicts"]=["next planned task"]
        with self.assertRaisesRegex(ValueError,"queue"):verify_queue(lock,receipt,source_commit="candidate",config_sha256="config",now=now)



    def test_actual_CPU_YOLO_clean_maps_cache_parity_and_rng(self):
        from ultralytics import YOLO
        from ue_framework.methods.base import BasePoisonGenerator
        from ue_framework.methods.tfss_clean_maps import build_maps
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        model=YOLO(str(ROOT/"configs/voc_yolov8n_20cls.yaml")).model.eval()
        for p in model.parameters():p.requires_grad_(False)
        class Probe:
            device=torch.device("cpu")
            target_class_id=14
            imgsz=128
            pad_value=.447
            context_inner=12
            context_outer=28
            channels=torch.tensor([0,6,10,29,41,68,77,86,89,112,116,125,128,133,140,167,173,184,220,229,237,238,239,240])
            surrogate=model
            _masks=LegacyKProtoTrainer._masks
            def _feature(self,x,grad):
                captured=[]
                handle=self.surrogate.model[21].register_forward_hook(lambda m,i,o:captured.append(o))
                try:self.surrogate(BasePoisonGenerator._letterbox_tensor(self,x))
                finally:handle.remove()
                return captured[0]
        folder=retained();path=folder/"clean.png"
        image=np.random.default_rng(4).random((96,112,3),dtype=np.float32)*.7+.15
        Image.fromarray(np.rint(image*255).astype(np.uint8)).save(path)
        image=np.asarray(Image.open(path)).astype(np.float32)/255
        ann=[P,DOG,HORSE]
        old_support=bare(S4Trainer,"bbox_half_overlap")._build_support(image.shape,ann)[0]
        new_support=full_support(image.shape,ann)
        before=torch.get_rng_state().clone();prior=random.getstate()
        old=CleanMapCache(folder/"half_cache",{"support_mode":"bbox_half_overlap","teacher":"CPU_random_YAML_no_download"})
        full=CleanMapCache(folder/"full_cache",{"support_mode":"bbox_full_target","teacher":"CPU_random_YAML_no_download"})
        # Unit fixture for the cache's 64-GiB production capacity check only.
        # Actual maps/model/native gradients and written cache bytes are real;
        # this fixture does not constitute remote resource admission evidence.
        from unittest.mock import patch
        with patch("ue_framework.methods.s4_tfss.shutil.disk_usage",return_value=SimpleNamespace(free=128*1024**3)):
            a,_=old.get(Probe(),[(image,ann,old_support)],[str(path)])
            b,_=full.get(Probe(),[(image,ann,new_support)],[str(path)])
            c,_=full.get(Probe(),[(image,ann,new_support)],[str(path)])
        for x,y,z in zip(a[0],b[0],c[0]):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
            torch.testing.assert_close(y,z,rtol=0,atol=0)
        self.assertEqual((old.misses,full.misses,full.hits),(1,1,1))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(prior,random.getstate())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_diagnostic_csv_schema_stable_with_and_without_overlap(self):
        from unittest.mock import patch
        obj=bare(S4FullTrainer,"bbox_full_target")
        x=torch.full((1,3,32,39),.5)
        obj.tfss_input=x
        metrics=[]
        for ann in ([P],[P,DOG]):
            s=torch.from_numpy(full_support((32,39,3),ann))
            obj.tfss_delta=s[None,None].expand_as(x)*.02
            obj.tfss_applied_weight=torch.ones(1,1,32,39)
            obj.s4_maps=[(s*0,s*0,s.bool())]
            obj.s4_spatial_rows=[{"image_id":"fixed","overlap_linf":0.}]
            with patch.object(S4Trainer,"_batch_loss",return_value=(torch.tensor(1.),{})):
                _,m=obj._batch_loss([(x[0].permute(1,2,0).numpy(),ann,s.numpy())],None)
            metrics.append(m)
        self.assertEqual(set(metrics[0]),set(metrics[1]))
        self.assertIsNone(metrics[0]["s4_overlap_RMS_delta"])
        self.assertAlmostEqual(metrics[1]["s4_overlap_RMS_delta"],.02,places=6)

    def test_optimizer_copy_changes_only_original_eligibility(self):
        import ast
        old=ast.parse((ROOT/"ue_framework/methods/s4_optimization.py").read_text())
        new=ast.parse((ROOT/"ue_framework/methods/s4_full_optimization.py").read_text())
        old.body[0]=new.body[0]
        expected_import=ast.parse("from .s4_tfss_full import original_training_eligible").body[0]
        old.body.insert(-1,expected_import)
        function=old.body[-1];function.name="optimize_s4_full"
        for node in ast.walk(function):
            if isinstance(node,ast.If) and ast.unparse(node.test)=="support.sum() > 10":
                node.test=ast.parse("original_training_eligible(im.shape, ann)",mode="eval").body
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))

    def test_historical_comparison_identity_and_same_results_zero_deltas(self):
        from scripts.s4_full_report import compare
        lock=json.loads((DOC/"S4_TFSS_FULL_LOCK.json").read_text())
        path=ROOT/lock["baseline"]["comparison_path"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),lock["baseline"]["comparison_sha256"])
        baseline=json.loads(path.read_text());old=baseline["metrics"]["S4"]
        self.assertEqual(old["checkpoint_sha256"],lock["baseline"]["last_sha256"])
        result=compare(baseline,copy.deepcopy(old))
        self.assertTrue(all(value==0 for value in result["aggregate_delta"].values()))
        self.assertAlmostEqual(result["scores"]["S4"]["S"],.935330,places=6)

    def test_baseline_identity_binds_training_and_export_files_separately(self):
        from unittest.mock import patch
        from scripts import run_s4_tfss_full as entry
        fixture=retained();old=fixture/"old";(old/"noise/checkpoints").mkdir(parents=True)
        (old/"noise/checkpoints/final.pt").write_bytes(b"training-state-identity-fixture")
        (old/"noise/global_params.pt").write_bytes(b"export-parameters-identity-fixture")
        (old/"materialization_manifest.json").write_text("[]")
        (old/"dataset_integrity.json").write_text("{}")
        weights=fixture/"victim/weights";weights.mkdir(parents=True)
        (weights/"last.pt").write_bytes(b"victim-identity-fixture")
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        comparison=dict(source_provenance=dict(last_sha256=digest(weights/"last.pt")),
                        training=dict(args=dict(project=str(fixture),name="victim")))
        (fixture/"comparison.json").write_text(json.dumps(comparison))
        config=fixture/"docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json";config.parent.mkdir(parents=True)
        config.write_bytes((ROOT/"docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json").read_bytes())
        lock=dict(baseline=dict(output=str(old),noise_checkpoint_relative_path="noise/checkpoints/final.pt",
            noise_sha256=digest(old/"noise/checkpoints/final.pt"),export_params_sha256=digest(old/"noise/global_params.pt"),
            manifest_sha256=digest(old/"materialization_manifest.json"),integrity_sha256=digest(old/"dataset_integrity.json"),
            comparison_path="comparison.json",comparison_sha256=digest(fixture/"comparison.json"),last_sha256=digest(weights/"last.pt")))
        with patch.object(entry,"ROOT",fixture):
            self.assertEqual(entry.baseline_identity(lock),comparison)
            wrong=copy.deepcopy(lock);wrong["baseline"]["noise_checkpoint_relative_path"]="noise/global_params.pt"
            with self.assertRaisesRegex(ValueError,"Historical S4 identity changed"):entry.baseline_identity(wrong)
            (old/"noise/global_params.pt").write_bytes(b"changed-export-fixture")
            with self.assertRaisesRegex(ValueError,"Historical S4 identity changed"):entry.baseline_identity(lock)

    def test_spatial_summary_distinguishes_no_overlap_and_area_pooling(self):
        from scripts.s4_full_report import spatial_summary
        rows=[]
        for ann in ([P,DOG],[P]):
            mask=full_support((32,39,3),ann)
            row=spatial_diagnostics(np.full((32,39,3),.5),mask[:,:,None]*np.ones((32,39,3))*.02,ann)
            row["baseline_S4_spatial"]=spatial_diagnostics(np.full((32,39,3),.5),mask[:,:,None]*np.ones((32,39,3))*.01,ann)
            rows.append(row)
        summary=spatial_summary(rows)
        self.assertEqual(summary["with_GT_overlap"]["images"],1)
        self.assertEqual(summary["without_GT_overlap"]["images"],1)
        self.assertAlmostEqual(summary["all_target_images"]["pooled_overlap_RMS_delta"],.02)
        self.assertAlmostEqual(summary["baseline_S4"]["pooled_overlap_RMS_delta"],.01)


if __name__=="__main__":unittest.main()
