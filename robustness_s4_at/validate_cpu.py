"""Real native YOLO/VOC CPU PGD10, SGD and controlled checkpoint-continuation probes."""
import argparse
from copy import deepcopy
from pathlib import Path
import random
import traceback
import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import ModelEMA
from ue_framework.s4_amp_observer import tree_hash
from .common import ROOT, protocol, native_identity, atomic, sha, now
from .data import VOC_NAMES
from .trainer import ATTrainer
from .pgd import PGDAttack, gt_identity, AttackConfig
from .checkpoint import capture, restore, save_once


def make_model(args):
    model = DetectionModel(str(ROOT / 'configs/voc_yolov8n_20cls.yaml'), nc=20, verbose=False)
    model.args = args
    model.train()
    model.criterion = model.init_criterion()
    return model


def native_optimizer(holder):
    holder.optimizer = DetectionTrainer.build_optimizer(holder, holder.model, name='SGD', lr=.01,
                                                        momentum=.937, decay=.0005)
    holder.scaler = torch.amp.GradScaler('cuda', enabled=False)
    holder.ema = ModelEMA(holder.model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(holder.optimizer, lambda e: (1 - np.cos(e * np.pi / 200)) / 2 * (.01 - 1) + 1)
    scheduler.last_epoch = -1  # native _setup_train sets this before the first epoch
    return scheduler


def new_holder(args, config, out, model=None):
    t = object.__new__(ATTrainer)
    t.args, t.device, t.stride = args, torch.device('cpu'), 32
    t.model = model if model is not None else make_model(args)
    t.validation_only, t.at_batches, t.epoch = True, 0, 0
    t.at_identity = dict(run_id='CPU_FIXTURE_AT' + str(config.epsilon_pixels), fixture=True)
    t.at_root = out
    t.attack = PGDAttack(config, 'cpu', seed=0)
    return t


def outer_step(t, batch):
    loss, parts = t.model(batch)
    loss.sum().backward()
    DetectionTrainer.optimizer_step(t)
    return float(loss.sum().detach()), {k: float(v) for k, v in parts.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(state='RUNNING', started_UTC=now(), device='cpu', official_metrics_disabled=True,
                  artifact_ingest_disabled=True, cleanup_policy='retain_all', deleted_paths=[],
                  actual_VOC_train_fixtures=True, production_GPU_readiness=False)
    try:
        torch.set_num_threads(1)
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        report['native_identity'] = native_identity()
        p = protocol()
        hyp = get_cfg(overrides=dict(p['train_args'], imgsz=160, batch=2, workers=0, cache=False,
                                     amp=False, plots=False, device='cpu'))
        fixture = __import__('json').loads((args.fixtures / 'fixture_identity.json').read_text())
        for row in fixture['records']:
            for k in ('image', 'label'):
                if sha(row[k + '_path']) != row[k + '_sha256']:
                    raise ValueError('VOC fixture bytes changed')
        ds = YOLODataset(img_path=str(args.fixtures / 'images/train'), imgsz=160,
                         batch_size=2, augment=True, hyp=hyp, cache=False, rect=False,
                         stride=32, data=dict(names=dict(enumerate(VOC_NAMES)), nc=20, channels=3))
        raw = YOLODataset.collate_fn([ds[0], ds[1]])
        report['native_pipeline'] = dict(dataset=type(ds).__name__, augmentation=str(ds.transforms),
                                         uint8_shape=list(raw['img'].shape), actual_ids=raw['im_file'],
                                         GT_count=int(raw['cls'].numel()),
                                         classes=sorted(set(raw['cls'].long().flatten().tolist())),
                                         native_augmentation_before_AT=True, validation_forwards=0,
                                         fixture_imgsz=160, fixture_batch=2, fixture_workers=0,
                                         formal_imgsz=640, formal_batch=36, formal_workers=16)
        report['budgets'] = []
        for eps, step in ((4, 1), (8, 2)):
            torch.manual_seed(0)
            config = AttackConfig(eps, step)
            t = new_holder(hyp, config, args.out / f'AT{eps}')
            scheduler = native_optimizer(t)
            params_before = tree_hash(t.model.state_dict())
            optimizer_before = tree_hash(t.optimizer.state_dict())
            next(v for v in t.model.parameters() if v.requires_grad).grad = torch.full_like(
                next(v for v in t.model.parameters() if v.requires_grad), .001)
            gradients_before = tree_hash({n: v.grad for n, v in t.model.named_parameters()})
            global_rng = torch.get_rng_state().clone()
            old_gt = gt_identity(raw)
            batch = t.preprocess_batch(deepcopy(raw))
            if params_before != tree_hash(t.model.state_dict()) or optimizer_before != tree_hash(t.optimizer.state_dict()):
                raise ValueError('PGD changed weights, BN state, or optimizer')
            if old_gt != gt_identity(batch) or not torch.equal(global_rng, torch.get_rng_state()):
                raise ValueError('GT or outer RNG changed in PGD')
            if tree_hash({n: v.grad for n, v in t.model.named_parameters()}) != gradients_before:
                raise ValueError('PGD contaminated existing accumulated parameter gradients')
            t.optimizer.zero_grad(set_to_none=True)
            bn_before = {n: v.clone() for n, v in t.model.named_buffers() if n.endswith('num_batches_tracked')}
            scheduler.step()
            loss, parts = outer_step(t, batch)
            if params_before == tree_hash(t.model.state_dict()) or t.ema.updates != 1:
                raise ValueError('Native outer SGD/EMA update was not reached')
            if any(int(dict(t.model.named_buffers())[n]) != int(v) + 1 for n, v in bn_before.items()):
                raise ValueError('Outer BN must update exactly once, not PGD10 times')
            identity = dict(run_id=t.at_identity['run_id'], protocol_sha256=sha(ROOT / 'robustness_s4_at/configs/PROTOCOL.json'),
                            source_files={p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / 'robustness_s4_at').glob('*.py')})
            payload = capture(t.model, t.optimizer, scheduler, t.scaler, t.attack, identity, 1, t.ema)
            payload.update(at_batches=t.at_batches, actual_updates=1, legal_amp_skips=0, best_fitness=0.)
            checkpoint = args.out / f'AT{eps}_epoch001_fullstate.pt'
            save_once(checkpoint, payload)
            u = new_holder(hyp, config, args.out / f'AT{eps}_resume')
            second_scheduler = native_optimizer(u)
            loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
            u.at_identity, u.resume_fullstate, u.epochs, u.train_loader = identity, checkpoint, 200, None
            u.scheduler = second_scheduler
            ATTrainer.resume_training(u, None)
            if u.start_epoch != 1:
                raise ValueError('Production Trainer resume did not consume the completed epoch')
            if tree_hash(t.model.state_dict()) != tree_hash(u.model.state_dict()) or tree_hash(t.optimizer.state_dict()) != tree_hash(u.optimizer.state_dict()):
                raise ValueError('Same-run restore did not recover model/optimizer exactly')
            left = t.preprocess_batch(deepcopy(raw)); right = u.preprocess_batch(deepcopy(raw))
            if not torch.equal(left['img'], right['img']):
                raise ValueError('Restored PGD RNG changed the next adversarial input')
            scheduler.step(); second_scheduler.step()
            left_loss, _ = outer_step(t, left); right_loss, _ = outer_step(u, right)
            if left_loss != right_loss or tree_hash(t.model.state_dict()) != tree_hash(u.model.state_dict()) \
                    or tree_hash(t.optimizer.state_dict()) != tree_hash(u.optimizer.state_dict()) \
                    or tree_hash(t.ema.ema.state_dict()) != tree_hash(u.ema.ema.state_dict()):
                raise ValueError('Controlled native next update differs after restore')
            rejected = False
            try:
                restore(loaded, u.model, u.optimizer, second_scheduler, u.scaler, u.attack,
                        dict(identity, run_id='DIFFERENT_ARM'))
            except ValueError:
                rejected = True
            if not rejected:
                raise ValueError('Cross-arm checkpoint restore was allowed')
            report['budgets'].append(dict(epsilon_pixels=eps, step_pixels=step,
                native_PGD10=t.at_latest, outer_detection_loss=loss, loss_parts=parts,
                native_SGD_and_EMA_updates=2, inner_model_and_optimizer_exact=True,
                outer_BN_updates_per_forward=1, controlled_next_update_restore_exact=True,
                cross_arm_restore_rejected=True, checkpoint_sha256=sha(checkpoint),
                input_RNG_and_GT_exact=True))
        # Empty-label batches keep native background classification supervision.
        empty = deepcopy(raw)
        empty.update(cls=torch.empty(0, 1), bboxes=torch.empty(0, 4), batch_idx=torch.empty(0))
        t = new_holder(hyp, AttackConfig(4, 1), args.out / 'empty_GT')
        t.preprocess_batch(empty)
        report['valid_empty_GT_background_loss_supported'] = True
        report['cuda_initialized'] = torch.cuda.is_initialized()
        if report['cuda_initialized']:
            raise ValueError('CPU validation unexpectedly initialized CUDA')
        report.update(state='PASSED_CPU_NATIVE_COMPONENT', completed_UTC=now(),
                      CUDA_smoke='PENDING_EXISTING_COCO_GPU_JOB',
                      scientific_review='NOT_CALLED_BEFORE_CUDA', formal_run_allowed=False)
    except BaseException as exc:
        report.update(state='FAILED', error=repr(exc), traceback=traceback.format_exc(), completed_UTC=now())
        raise
    finally:
        atomic(args.out / 'report.json', report)
    print(report['state'])


if __name__ == '__main__':
    main()
