# CPU AMP contracts — 2026-10-18

Command: bundled local Python, `-B -m unittest discover -s tests -p test_s4_amp_observer.py -v`.

Result: **7 passed**, 5.300seconds. Tested: inherited native optimizer method; exact finite model/SGD/scaler/scheduler state equality; overflow skips actual update and halves scale; following finite update succeeds; accumulation is not counted as a skip;16 consecutive skips stop with retained evidence; nonfinite raw loss stops; frozen ceiling cannot silently change. Artifacts remain under `execution/cpu_amp_tests/`.

The first invocation used a module-style test name and resolved a third-party `tests` package; it ran no project tests and failed discovery. The corrected discovery command passed. Temporary Ultralytics settings were generated only inside this isolated worktree and are excluded from source/publication. No environment installation/change or GPU/model/data replay occurred.

These tests do not replace the required real CUDA Victim smoke or establish recovery of the original failing batch.
