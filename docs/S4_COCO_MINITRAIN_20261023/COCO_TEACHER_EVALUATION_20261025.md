# Fixed-last Teacher assessment — independently accepted

Original workflow teacher200 completed200 epochs with zero validation forwards.
Its separate read-only official COCO80 assessment evaluated all original5000
clean val2017 IDs. Source 3ce80a512157b46cdeca8ab52799bd01bf94dc76; checkpoint 9749a7b6ceeb7b322316bdd97dfdde68f11641eef59fe6ba620a5e76a9521e3f;
server acceptance 69655e8c89aa282a5ace5eff89b45412d9631f76409d269d778728c1df868b51.

Person AP50=0.711971060380, AP75=0.486437733439,
AP50–95=0.462561678095.
All80 macro AP50=0.395154267840, AP75=0.283989529772,
AP50–95=0.264909898204.

Actual pre-NMS tensors were CUDA0/index0. Native data batches139, maximum36,
last32, plus one native warmup. All80 per-class JSON/CSV rows and hashes were
independently checked. No checkpoint update/selection, validation statistics
for S4, AP tuning, or formal metrics from a smoke were allowed.
These are Teacher descriptive measurements from seed0, not S4 effectiveness,
replicated significance or causal evidence. Original S4 8/4/0 is unchanged.

The original72-image failed device-audit attempt remains FAILED. Its logs and
records remain retained. An additional scientific review identified that list
input bypassed batch36; its negative B1 verdict is retained. Production fixed
native text-source batching. A new72-image GPU smoke demonstrated two36 batches
and warmup, exact native raw/decoded outputs, actual CUDA0. Main closed B1 with
production evidence; no second/closing Teacher-evaluation reviewer was used.
The original serial controller automatically ran formal evaluation after READY.

GitHub API initially preserved the original3b3a0af code tree under a new commit
ID. A later ordinary nonforce push preserved the exact original commit
3b3a0af92d2ae2ab224879cd6dc296b68bbc9f35 in branch history (parent of5833f527),
without changing the tested tree or overwriting remote commits.

Accepted data was reused without downloads/extraction/full scans. All failed
and diagnostic artifacts were retained. Later calibration/Fourier/export and
matchedVictim phases are separately gated; this assessment does not establish
full Experiment A completion.
