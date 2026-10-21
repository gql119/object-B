# Code change list

All runtime code is additive in this branch. Original S4 source files remain byte-identical; hashes are in SOURCE_AUDIT.json.

| File | Change |
|---|---|
| `ue_framework/methods/s4_tfss_full.py` | opt-in full support / generator / inherited S4 losses / GT diagnostics |
| `ue_framework/methods/s4_full_optimization.py` | original optimizer copied with only sample-eligibility expression changed |
| `ue_framework/s4_full_contract.py` | one-variable config and predecessor/queue admission |
| `ue_framework/configs/exp_voc_person_s4_tfss_full_seed0_20261021.yaml` | only YAML support_mode differs from original S4 |
| `scripts/run_s4_tfss_full.py` | independent generation-export-audit-victim-evaluation entrypoint |
| `scripts/evaluate_s4_tfss_full.py` | first evaluation of completed200 only; no training |
| `scripts/s4_full_report.py` | VOC20/PUS/WCD/TCD3/class deltas and spatial distributions |
| `scripts/check_s4_tfss_full_cpu.py` | bounded CPU test replay and source-bound evidence |
| `tests/test_s4_tfss_full.py` | geometry/numeric-gradient/cache/export/identity/queue regressions |

Documentation and configuration records are under docs/S4_TFSS_FULL_20261021/. Local runtime, retained tests and continuation records are under execution/ and are not dataset/model publication artifacts.
