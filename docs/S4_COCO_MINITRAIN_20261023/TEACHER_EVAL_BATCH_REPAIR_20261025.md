# Teacher evaluation batch-sink repair

The one independent Teacher evaluation scientific review rejected candidate
52cb88e for B1: a list of 72 paths reaches Ultralytics' in-memory loader as one
batch despite the frozen `batch=36`. The retained smoke proves CUDA0 and exact
native predictions at batch72, but does not satisfy the formal batch contract.
Its failed scientific verdict, smoke outputs and the original exit1 smoke are
retained. Neither candidate has a formal READY registration.

Production now supplies an ordered text manifest to the unchanged native
LoadImagesAndVideos path, which consumes the frozen batch argument. The
separate same-checkpoint native reference constructs its own text manifest.
Both hooks record actual forward input batch sizes alongside pre-NMS tensor
device/index evidence. A forward exceeding the frozen batch size raises an
error before postprocessing. A replacement smoke must prove two planned
batches of 36, actual inputs at most36, CUDA0/index0, finite outputs and exact
independent raw/decoded predictions. Full validation remains the same5000
original images in the same order, with a final partial batch of32.

No model weights, prediction settings, category mapping, COCO metric settings,
training hyperparameters or Original S4 mathematics/budgets change. Smoke
metrics and ingest stay disabled. The main agent closes B1 with actual repaired
production evidence; there is no second scientific review.

Adjacent registration transport corrections use a shorter local pull root and
the actual RRCTL `pull_root/run_id/result` layout, and read the prediction
audit's existing `checkpoint_sha256` field. These changes do not compute a
scientific result or launch another controller.

At implementation time the replacement GPU smoke, B1 closure and formal
evaluation remain OPEN. Local CPU evidence is not CUDA validation.
