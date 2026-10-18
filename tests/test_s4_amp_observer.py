import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
import torch
from ue_framework.s4_amp_observer import AMPAuditObserver, tree_hash
from scripts.s4_victim_recovery import baseline_prefix_without_update_is_valid

ROOT=Path(__file__).resolve().parents[1]

def trainer():
    model=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():model.weight.fill_(1.)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
    return SimpleNamespace(model=model,optimizer=optimizer,scaler=torch.amp.GradScaler('cpu'),
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200),
        ema=None,epoch=0,device=torch.device('cpu'),loss=None,tloss=None)

def step(t,observer=None,multiplier=1.,overflow=False,accumulate=False):
    if observer:observer.on_train_batch_start(t)
    p=t.model.weight
    t.loss=(p-p.detach()).sum()*3e38 if overflow else p.sum()*multiplier
    t.tloss=t.loss.detach()
    t.scaler.scale(t.loss).backward()
    if not accumulate:
        t.scaler.unscale_(t.optimizer)
        torch.nn.utils.clip_grad_norm_(t.model.parameters(),max_norm=10.)
        t.scaler.step(t.optimizer);t.scaler.update();t.optimizer.zero_grad()
    if observer:observer.on_train_batch_end(t)

class AMPContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.output=ROOT/'execution/cpu_amp_tests'/uuid.uuid4().hex
        cls.output.mkdir(parents=True,exist_ok=False)

    def observer(self,t,name):
        o=AMPAuditObserver(self.output/name,debug_hashes=True)
        o.on_train_start(t);o.on_train_epoch_start(t)
        self.addCleanup(o.stream.close)
        return o

    def test_no_optimizer_override(self):
        source=ast.parse((ROOT/'ue_framework/s4_victim.py').read_text())
        cls=next(n for n in source.body if isinstance(n,ast.ClassDef))
        self.assertNotIn('optimizer_step',[n.name for n in cls.body if isinstance(n,ast.FunctionDef)])

    def test_finite_native_sequence_exact(self):
        reference,candidate=trainer(),trainer();o=self.observer(candidate,'finite')
        for _ in range(3):step(reference);step(candidate,o)
        for name in ('model','optimizer','scheduler','scaler'):
            self.assertEqual(tree_hash(getattr(reference,name).state_dict()),tree_hash(getattr(candidate,name).state_dict()))
        self.assertEqual((o.updates,o.skips),(3,0))
        self.assertIsNotNone(o.rows[-1]['post_native_clip_unscaled_gradient_norm'])

    def test_overflow_skip_backoff_then_recover(self):
        t=trainer();o=self.observer(t,'overflow');before=tree_hash(t.model.state_dict())
        step(t,o,overflow=True)
        self.assertEqual(o.updates,0);self.assertEqual(o.skips,1)
        self.assertEqual(tree_hash(t.model.state_dict()),before)
        self.assertEqual(t.scaler.get_scale(),32768.)
        self.assertTrue(all(p.grad is None for p in t.model.parameters()))
        self.assertIsNone(o.rows[-1]['post_native_clip_unscaled_gradient_norm'])
        self.assertTrue(o.rows[-1]['parameters_unchanged_on_skip'])
        step(t,o)
        self.assertEqual(o.updates,1);self.assertEqual(o.consecutive_skips,0)

    def test_accumulation_not_counted_as_skipped_update(self):
        t=trainer();o=self.observer(t,'accumulation');step(t,o,accumulate=True)
        self.assertEqual((o.attempts,o.skips,o.updates),(0,0,0))
        step(t,o);self.assertEqual((o.attempts,o.skips,o.updates),(1,0,1))

    def test_persistent_overflow_stops_at_frozen_limit(self):
        t=trainer();o=self.observer(t,'persistent')
        for _ in range(15):step(t,o,overflow=True)
        with self.assertRaisesRegex(FloatingPointError,'predeclared16'):step(t,o,overflow=True)
        self.assertEqual(o.consecutive_skips,16)
        self.assertTrue((o.directory/'failure.json').exists())

    def test_nonfinite_raw_loss_is_not_ignored(self):
        t=trainer();o=self.observer(t,'nonfinite_loss')
        with self.assertRaisesRegex(FloatingPointError,'Nonfinite raw loss'):
            step(t,o,multiplier=float('nan'))
        self.assertEqual(o.updates,0)

    def test_policy_cannot_be_relaxed_silently(self):
        with self.assertRaises(ValueError):AMPAuditObserver(self.output/'bad',max_consecutive_skips=17)

    def test_zero_update_smoke_prefix_only_accepts_logged_native_baseline_skips(self):
        rows=[{'native_scaler_nonfinite_skip':True,'backoff':True,'optimizer_update_executed':False} for _ in range(5)]
        self.assertTrue(baseline_prefix_without_update_is_valid('native_S0',rows))
        self.assertTrue(baseline_prefix_without_update_is_valid('repaired_S0',rows))
        self.assertFalse(baseline_prefix_without_update_is_valid('real_S4',rows))
        self.assertFalse(baseline_prefix_without_update_is_valid('native_S0',[]))
        broken=list(rows);broken[-1]=dict(rows[-1],backoff=False)
        self.assertFalse(baseline_prefix_without_update_is_valid('native_S0',broken))

if __name__=='__main__':unittest.main()
