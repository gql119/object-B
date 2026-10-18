# Stage B entrypoint import repair

Change: source410212b used an absent `resource_guard` export from `scripts.run_s4_tfss_pipeline`. The inherited active API is `resource_gate()`. Change only the import and call in `scripts/s5_layer_probe.py`; preserve the existing gate, scientific formulas, fixed32 manifest, five-step/EOT2 scope and all45 original source bytes.

Result: unique `s5_p3_p5_train32_layerprobe_seed0_20261018_v1` failed exit1 at2026-10-18T17:43:26Z before model setup/optimizer updates. It is terminal and will never be relaunched. RRCTL inspect and diagnostic pull succeeded; six small artifacts retained under local s5_small_20261018/<RunID>/diagnostics/b161d3bf8eec499a8ee807044a9b3d9a. Console SHA2561d85880f51ef925b90072e9c5012d6d1cb0aaff37b13d7a102cbdabcfb4d648f, 2331bytes. Exception: ImportError cannot import name resource_guard. This is an entrypoint defect, not P3/P5 or method evidence.

Validation: actual scientific dependency imports/exports8 PASS; compilation PASS; targeted pytest `tests/test_s5_retention_layer.py tests/test_s4_metric_delta_schema.py`: 5 passed in5.83s; all45 canonical file SHA256 exact. Current-source GPU work remains NOT_RUN.

Next: commit repair and create a fresh unique v2 with unchanged scope. Fresh resource admission/preflight and four-path absence mandatory. Preserve v1 failed source/output/control, do not call a reviewer before current CUP candidate GPU smoke.
