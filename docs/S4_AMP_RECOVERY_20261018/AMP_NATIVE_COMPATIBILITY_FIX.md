# Native Victim AMP compatibility repair

The failed wrapper checked AMP-scaled gradients before native unscale/skip/backoff. The repair removes its `optimizer_step` override entirely: `S4VictimTrainer.optimizer_step` is inherited from the installed `DetectionTrainer`. No second unscale, optimizer call, gradient modification or custom scaler implementation exists.

`s4_amp_observer.py` observes public SGD pre/post hooks and trainer batch callbacks. The post-hook counts actual SGD updates, including distinguishing accumulation-only batches from scaler skips. A native scale reduction with no actual SGD call records a nonfinite skip/backoff event. The pre-hook only reads gradients after native unscale and clipping and records that norm with its stage label. Loss/model/EMA/optimizer finiteness checks and checkpoint retention remain safety instrumentation. No diagnostic consumes RNG or changes an input tensor.

Before CUDA, the consecutive native-skip ceiling is fixed to16: sixteen successive halvings span the default65536→1 scale ladder. Reaching the ceiling records model/optimizer/scaler/scheduler/RNG and actual input batch evidence and stops. This engineering ceiling does not assert that all isolated skips are benign: subsequent actual updates and finite states must be demonstrated. Accumulation-only batches do not count as skips.

The user authorizes this implementation repair, actual-path verification and victim-only recovery. Prior S4's single independent review remains `not_evaluable` with main-agent evidence closure; no new/closing reviewer is requested or claimed. Current repair CPU/CUDA evidence will be explicit. TFSS math, maps, S0 losses, source45, data and all formal victim hyperparameters remain frozen.
