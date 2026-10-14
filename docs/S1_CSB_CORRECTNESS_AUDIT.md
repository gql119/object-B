# S1 correctness audit

Stage0 source/config/checkpoint/metrics provenance is located. Historical-source vs restored-source 5-step GPU equivalence passed in `s1_parent_equivalence_train16_seed0_20261014_v2`: max_loss_difference=2.980232238769531e-8, max_tensor_difference=3.725290298461914e-9, tolerance=1e-6. Evidence binds actual source imported in separate processes, checkpoint/prototype/config and same 16 train ids. Tensor comparison covers target/utility gradients, total gradient, poison and Fourier updates for every step/EOT. v1 path-binding failure retained; v2 exited0/rrctl completed.

CSB masked normalization, no-NT fallback/error distinction and PNG per-pixel quantized bounds: 3 local tests passed. CSB spatial GPU audit, projection equivalence, exact resume, 5-step paired diagnosis and independent review are not yet complete. No formal S1 or victim is running.
