# S0 / failed S4 / repaired S4 Victim paths

S0's retained matched launcher constructs random-YAML `YOLO` and calls `model.train(**train_args)`, using the installed native DetectionTrainer. The archived Victim specification freezes batch36/workers16/AMP/SGD/LR/scheduler/augmentation/seed0. S0's contemporaneous installed Trainer source hash is unknown; the original checkpoint's version and arguments must be re-read remotely. The previously captured environment records PyTorch2.7.1+cu128 / Ultralytics8.4.150. Those are prior records until the new audit re-verifies them.

The failed S4 subclass introduced a six-line optimizer override: finite raw-state checks, fatal pre-unscale gradient check, then `super().optimizer_step()`. The gradient check prevents native GradScaler skip/backoff. S4 does not require a different Victim optimizer: its scientific intervention is the training-image perturbation.

The repair deletes that override. `S4VictimTrainer.optimizer_step is DetectionTrainer.optimizer_step` is a required runtime smoke assertion. Retained subclass functionality only guards nonfinite epoch/model states and retains additional checkpoint copies; the observer uses public hooks and callbacks. It never unscales, clips, modifies gradients, calls SGD.step or changes scale. SGD pre/post hooks measure only the actual update calls made by native GradScaler.

Full implementation diff is in `S0_S4_VICTIM_FULL.diff`. Native source excerpts and original S0 checkpoint metadata will be retained by the read-only CPU audit. Actual CUDA baseline equivalence is NOT_RUN while RRCTL is unreachable. Its comparison uses identical S0 inputs and initialization for the native and repaired paths, then a separate actual S4 batch trial; clean/poison input differences are not treated as Trainer differences.

Training bytes/GT remain identical: per-file links under new real directories isolate YOLO cache metadata inside the new recovery output. Directory symlinks are intentionally avoided because native YAML path resolution could redirect caches to the original dataset. No transform, image order, normalization or training parameter is changed by this storage view.
