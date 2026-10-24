"""Strict train-only COCO Teacher around the original S4/native training arithmetic.

This module does not launch a remote run or supply a scientific-review gate. The
serial workflow must bind its source, data and configuration before using it.
Native final-epoch/final-eval validation is explicitly excluded for the Teacher.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import torch
import yaml
from ue_framework.s4_victim import S4VictimTrainer


class _EmptyValidationDataset:
    labels = ()
    im_files = ()

    def __len__(self):
        return 0


class _ForbiddenValidationLoader:
    dataset = _EmptyValidationDataset()
    num_workers = 0

    def __len__(self):
        return 0

    def __iter__(self):
        raise RuntimeError('COCO Teacher must never iterate a validation loader')


class _ForbiddenValidator:
    def __init__(self):
        self.metrics = SimpleNamespace(keys=[])
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError('COCO Teacher validation forward is forbidden')


def teacher_overrides(config, data_yaml, output):
    """Copy the already frozen native victim protocol; no AP-based selection."""
    v = config['victim']
    out = Path(output).resolve()
    init = Path(v['init'])
    if not init.is_absolute():
        init = Path(__file__).resolve().parents[1]/init
    return dict(model=str(init.resolve()), data=str(Path(data_yaml).resolve()),
                project=str(out.parent), name=out.name, exist_ok=True,
                epochs=v['epochs'], batch=v['batch'], workers=v['workers'], imgsz=v['imgsz'],
                optimizer=v['optimizer'], cos_lr=v['cos_lr'], close_mosaic=v['close_mosaic'],
                cache=v['cache'], amp=v['amp'], save_period=v['save_period'],
                lr0=v['lr0'], lrf=v['lrf'], momentum=v['momentum'],
                weight_decay=v['weight_decay'], seed=v['seed'],
                pretrained=False, patience=0, resume=False, val=False,
                device=v['device'], plots=False, classes=None, single_cls=False,
                fraction=1.0, rect=False, task='detect')


class COCOTrainOnlyTeacher(S4VictimTrainer):
    """Keep native train loop/loss/SGD/AMP; exclude all validation data and calls.

    ``cpu_probe`` is only a local validation route, never a formal configuration.
    No old training output can be reopened, incremented or resumed by this class.
    """
    def __init__(self, *args, cpu_probe=False, **kwargs):
        overrides = kwargs.get('overrides')
        if args or not isinstance(overrides, dict) or set(kwargs)-{'overrides', '_callbacks'}:
            raise ValueError('Use explicit frozen overrides and a fresh bound output')
        allowed = {'model','data','project','name','exist_ok','epochs','batch','workers','imgsz',
                   'optimizer','cos_lr','close_mosaic','cache','amp','save_period','lr0','lrf',
                   'momentum','weight_decay','seed','pretrained','patience','resume','val',
                   'device','plots','classes','single_cls','fraction','rect','task'}
        if set(overrides) != allowed:
            raise ValueError('Teacher arguments must use only the frozen production schema')
        name = overrides['name']
        if not isinstance(name, str) or name in {'', '.', '..'} or '/' in name or '\\' in name or ':' in name:
            raise ValueError('Teacher output name must be one new path component')
        if not Path(overrides['project']).is_absolute():
            raise ValueError('Teacher output parent must be absolute and workflow-bound')
        self.cpu_probe = cpu_probe
        self.teacher_audit = dict(validation_loader_requests_blocked=0,
                                  native_validate_dispatches_blocked=0,
                                  final_eval_dispatches_blocked=0,
                                  validation_forwards=0, EMA_forwards=0,
                                  fitness_kind='negative_native_training_loss; not AP',
                                  formal_primary_checkpoint='last@200', automatic_resume_allowed=False)
        if overrides.get('val') is not False or overrides.get('resume') is not False or overrides.get('pretrained') is not False:
            raise ValueError('Teacher requires val=False, resume=False, pretrained=False')
        if Path(overrides['model']).suffix != '.yaml':
            raise ValueError('Teacher initialization must be random YAML, never .pt')
        target = Path(overrides['project']).resolve()/overrides['name']
        if target.exists():
            raise FileExistsError('Teacher output collision; no overwrite/resume/increment')
        if cpu_probe:
            if overrides.get('device') != 'cpu' or overrides.get('epochs') != 1:
                raise ValueError('CPU probe must be explicitly one CPU epoch')
        else:
            frozen = dict(epochs=200, batch=36, workers=16, imgsz=640, optimizer='SGD',
                          cos_lr=True, close_mosaic=100, cache=True, amp=True, seed=0,
                          lr0=.01, lrf=.01, momentum=.937, weight_decay=.0005, patience=0,
                          device='0', save_period=10, classes=None, single_cls=False,
                          fraction=1.0, rect=False, task='detect', exist_ok=True)
            for k, value in frozen.items():
                if overrides.get(k) != value:
                    raise ValueError(f'Frozen COCO Teacher argument changed: {k}')
        super().__init__(**kwargs)

    def get_dataset(self):
        # Do not call check_det_dataset: this data file intentionally has no val
        # split and no download stanza. Only the audited clean train tree exists
        # in the returned dataset, so native setup cannot scan clean val2017.
        p = Path(self.args.data)
        data = yaml.safe_load(p.read_text(encoding='utf-8'))
        if set(data) != {'path', 'train', 'val', 'nc', 'names'} or data['val'] is not None:
            raise ValueError('Teacher YAML must expose only train and a null val')
        names = data['names']
        if isinstance(names, list):
            names = dict(enumerate(names))
        if data['nc'] != 80 or set(names) != set(range(80)) or names[0] != 'person':
            raise ValueError('Teacher requires COCO80 and Person0')
        mapping_path = Path(__file__).resolve().parents[1]/'docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json'
        mapping_bytes = mapping_path.read_bytes()
        if hashlib.sha256(mapping_bytes).hexdigest() != '72d8e90134dade3cc3dc6f7a2b51df2c5758a60e315165a255f15258d26a8814':
            raise ValueError('Canonical COCO category mapping bytes changed')
        expected = {r['yolo_class_id']:r['name'] for r in json.loads(mapping_bytes)['mapping']}
        if names != expected:
            raise ValueError('Teacher names/order differ from the audited official COCO80 mapping')
        root = Path(data['path']).resolve()
        if data['train'] != 'images/train' or not (root/'images/train').is_dir():
            raise ValueError('Teacher must consume the audited clean train split')
        return dict(path=root, train=str(root/'images/train'), val=None,
                    names=names, nc=80, channels=3, yaml_file=str(p.resolve()))

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if mode == 'val':
            self.teacher_audit['validation_loader_requests_blocked'] += 1
            return _ForbiddenValidationLoader()
        if mode != 'train' or str(dataset_path) != self.data['train']:
            raise ValueError('Unexpected Teacher data route')
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def _build_train_pipeline(self):
        if self.args.batch != 36 or self.batch_size != 36:
            raise ValueError('No native OOM batch reduction for the matched Teacher')
        return super()._build_train_pipeline()

    def get_validator(self):
        return _ForbiddenValidator()

    def label_loss_items(self, loss_items=None, prefix='train'):
        if prefix == 'val':
            if loss_items is not None:
                raise RuntimeError('Teacher cannot report invented validation losses')
            return []
        return super().label_loss_items(loss_items, prefix)

    def _setup_train(self):
        super()._setup_train()
        self.metrics = {}
        self.teacher_audit['train_image_count'] = len(self.train_loader.dataset)
        if not self.cpu_probe and len(self.train_loader.dataset) != 25000:
            raise ValueError('Formal Teacher requires all 25K training images')
        self.teacher_audit['data_route'] = self.data['train']
        self.teacher_audit['validation_image_count'] = len(self.test_loader.dataset)
        self._teacher_forward_guard = self.ema.ema.register_forward_pre_hook(self._forbid_ema_forward)

    def _forbid_ema_forward(self, module, args):
        self.teacher_audit['EMA_forwards'] += 1
        raise RuntimeError('Teacher EMA must not perform inference/validation')

    def save_model(self):
        # Audit hooks must never be deep-copied/serialized into native EMA
        # checkpoints. Remove only this passive hook around the unchanged S4
        # save operation; restore it on the live EMA on every exit path.
        handle = getattr(self, '_teacher_forward_guard', None)
        if handle is not None:
            handle.remove()
            self._teacher_forward_guard = None
        try:
            return super().save_model()
        finally:
            if handle is not None:
                self._teacher_forward_guard = self.ema.ema.register_forward_pre_hook(self._forbid_ema_forward)

    def validate(self):
        # Native train loop dispatches here on final_epoch even with val=False.
        # Use the same training-loss fallback concept as native validate(), but
        # never construct or execute validation. This fitness never drives the
        # optimizer/scheduler, patience is disabled, and only last@200 is used.
        self.teacher_audit['native_validate_dispatches_blocked'] += 1
        self._finite()
        if not torch.is_tensor(self.loss) or not torch.isfinite(self.loss).all():
            raise FloatingPointError('Missing/nonfinite native Teacher training loss')
        fitness = -float(self.loss.detach().sum().cpu())
        if self.best_fitness is None or self.best_fitness < fitness:
            self.best_fitness = fitness
        return {}, fitness

    def final_eval(self):
        # Native final_eval also strips last/best optimizer state. Keep original
        # checkpoints, full-state snapshots and native SGD/RNG evidence intact.
        self.teacher_audit['final_eval_dispatches_blocked'] += 1
        if self.validator.calls or self.teacher_audit['EMA_forwards']:
            raise RuntimeError('Teacher reached a forbidden validation path')
        self._finite()
        self.teacher_audit.update(validation_forwards=0, cpu_probe=self.cpu_probe,
                                  native_optimizer_step_inherited=True,
                                  checkpoints_not_stripped=True,
                                  official_metrics_disabled=True)
        if self.cpu_probe:
            self.teacher_audit['artifact_ingest_disabled'] = True
        (Path(self.save_dir)/'TEACHER_TRAIN_ONLY_AUDIT.json').write_text(
            json.dumps(self.teacher_audit, indent=2), encoding='utf-8')
