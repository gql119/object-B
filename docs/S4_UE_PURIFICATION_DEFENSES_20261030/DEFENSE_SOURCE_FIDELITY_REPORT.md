# Defense source-fidelity audit: OP, AVATAR, D-VAE, LE

Date: 2026-10-30
Scientific parent: SMIC-Det Original S4 = Safe-Soft05 + TFSS-Utility
Target task: VOC20 detection, protected class Person (class id 14)
Evidence index: [`UPSTREAM_SOURCE_INDEX.json`](UPSTREAM_SOURCE_INDEX.json)

## Scope and evidence rules

The audit pinned the MI-UE repository and each defense repository to a Git commit, then read the listed paper, source entry points, configurations, and README/checkpoint statements. Forty-three text files are recorded by upstream Git blob SHA in the index. Local source copies are for inspection only; CRLF-to-LF normalization, when present, applies only to those local analysis copies. No author implementation, model, checkpoint, or dataset was copied into this project.

The MI-UE Table 5 is a CIFAR-10 image-classification comparison. It is context for why these defenses are being considered, not evidence for VOC detection. The MI-UE README's paper hyperlink resolves to a different 2022 paper; the specified MI-UE paper is cited below by its arXiv record. Its Table 5 is not used to choose detection settings. [MI-UE paper](https://arxiv.org/abs/2603.03725), [MI-UE source at pinned commit](https://github.com/hala64/mi-ue/tree/4cd387978c08b0c85eed9afc0b849d0b8777691c)

## OP — Orthogonal Projection

- **Paper:** *What Can We Learn from Unlearnable Datasets?*, NeurIPS 2023. [Proceedings record](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ee5bb72130c332c3d4bf8d231e617506-Abstract.html), [paper](https://arxiv.org/abs/2305.19254).
- **Author code:** [pinned repository](https://github.com/psandovalsegura/learn-from-unlearnable/tree/866d1dab6420bb1ea99796c8a263f3b0562e23e9); inspected `orthogonal_projection_step_1.py`, `orthogonal_projection_step_2.py`, `utils_projection.py`, `datasets.py`, and `constants.py`.
- **Mechanism actually implemented:** train a linear classifier using one classification label per image; interpret each classifier row as a recovered class-wise direction; normalize directions; form an orthogonal basis with QR; sequentially subtract each projection from flattened image vectors. The paper and README emphasize class-wise perturbations. This is not a generic image denoiser or high-pass filter.
- **Parameters disclosed in the checked code:** step 1 defaults to 20 epochs, SGD learning rate 0.1, batch size 1024; its scheduler drops at half and three-quarters of the configured epoch count. The code is set up around small image classification (the CIFAR path reshapes a direction as 3×32×32). These are author-code defaults for that setup, not VOC values. The paper/repository do not define a detector-specific projection rank, multi-object target aggregation, or VOC20 setting; those items are **NOT_DISCLOSED**.
- **Data/model needs:** protected classification images with one class label per image and a trained linear classifier for direction recovery. No VOC detector projection checkpoint or detection training recipe is present in the inspected project.
- **Detection gap:** one image may contain several classes and instances, including Person and non-target objects. Choosing a per-image label, pooling instance directions, or projecting only a region each changes the algorithm. No one of those mappings follows automatically from the classifier code.
- **S4 adaptation status:** **not faithful as a direct port; `SCIENTIFIC_ADAPTATION_REQUIRED`.** A detection definition would have to explain how directions are estimated and applied with multi-class GT and how background pixels are handled, then validate the definition independently. No such definition is asserted here.
- **License:** no root license file was found at the pinned upstream tree. Upstream implementation code was not copied.

## AVATAR — diffusion-based sanitization

- **Paper:** *The Devil's Advocate: Shattering the Illusion of Unexploitable Data using Diffusion Models*, SaTML 2024. [Paper](https://arxiv.org/abs/2303.08500), [author repository at pinned commit](https://github.com/hmdolatabadi/AVATAR/tree/978df2e1e4fef6ea0456b12a7fc44f052010f182).
- **Author entry points:** `AVATAR_sanitization.py`, `AVATAR_evaluation.py`, the DDPM/guided/SDE/ODE runners, and the CIFAR-10/ImageNet configs.
- **Mechanism actually implemented:** use a pretrained, domain-specific diffusion/score model; perturb the protected image forward to a selected noise level and reverse-denoise it. The denoising level trades removal of the unlearnable perturbation against loss of clean semantic information. The paper evaluates candidate levels rather than defining a universal “100 steps” setting.
- **Parameters and model scope disclosed:** the checked CIFAR-10 Score-SDE config is 32×32; the checked ImageNet DDPM config is 256×256. The paper discusses model/data-specific checkpoints and a timestep selection sweep. Those settings do not establish a VOC20-at-640 checkpoint or optimal timestep. A filename containing “100” is not accepted as proof that 100 is the Table 5 setting.
- **Data/model needs:** pretrained diffusion weights matched to the data distribution. The source tree contains code/configuration and license files, not an independently verified VOC20 checkpoint. The upstream README points to domain-specific public models; none was downloaded or loaded in this audit.
- **Detection gap:** pure pixel denoising can preserve tensor dimensions and bbox coordinates structurally, but it cannot by itself prove that a small object, boundary, or object identity survives at the same position. Resizing a 32×32 or 256×256 model input to/from VOC 640 changes spatial detail; no VOC detector validation or GT-sync protocol is published in the inspected source.
- **S4 adaptation status:** no YOLO adapter exists. The immediate status is **`BLOCKED_DEPENDENCY`** until a legally usable, distribution-compatible checkpoint and fixed geometry protocol are identified. If adaptation changes model resolution or denoising semantics, the result must be labeled as an adaptation, not a strict AVATAR reproduction.
- **License:** upstream contains `LICENSE` and `LICENSE_DIFFPURE`; component-specific terms must be reviewed before code reuse. No code or weights were copied.

## D-VAE — disentangled VAE purification

- **Paper:** *Purify Unlearnable Examples via Rate-Constrained Variational Autoencoders*, ICML 2024. [PMLR record](https://proceedings.mlr.press/v235/yu24m.html), [paper](https://arxiv.org/abs/2405.01460), [pinned source](https://github.com/yuyi-sd/D-VAE/tree/50b9ca08135301f47e678eddfcb04fea47a850b9).
- **Author entry points:** `main_vae_disentangle.py`, `main.py`, `trainer.py`, `evaluator.py`, `dataset.py`, model files, and CIFAR-10/ImageNet-mini configs.
- **Mechanism actually implemented:** class-conditioned embeddings feed a decoder that estimates the unlearnable perturbation from the latent representation; rate-constrained VAE objectives separate recoverable image content from the perturbation. Purification is two-stage: a first pass obtains a coarse purified image, then a second trained stage refines/reconstructs it. The two stages are part of the method and cannot be silently collapsed into one generic autoencoder pass.
- **Parameters disclosed for the inspected examples:** CIFAR-10 config specifies 10 classes, 60 epochs, latent dimension 256, Adam at 0.005, and train/eval batch 128. ImageNet-mini config uses 100 classes and a different model stem setting. These are dataset examples, not a VOC20 prescription. The paper uses rate/KLD settings and reconstruction-quality evaluation to choose operating points; no validated VOC20 640×640 pair of rate targets is published. Exact detector-condition mapping is **NOT_DISCLOSED**.
- **Data/model needs:** train the VAE/disentangling stages on the intended protected-data distribution or load a model trained for that domain. The repository links CIFAR experiments; no compatible VOC checkpoint was present or downloaded. The inspected code is classification-oriented and includes CUDA-specific assumptions.
- **Detection gap:** a VOC image can have multiple instance classes; replacing that set with one class embedding is invalid. A spatial class map or per-instance condition is a new detection adaptation and must define overlaps, background, empty-GT images, and how both purification stages use those conditions. Scaling to 640 must also audit small-object and border fidelity.
- **S4 adaptation status:** **`SCIENTIFIC_ADAPTATION_REQUIRED`**. The necessary multi-instance conditioning and rate-selection protocol are not derivable from the CIFAR configuration. No adaptation is implemented here.
- **License:** MIT file is present at the pinned commit; license presence does not supply the missing method definition, data, or checkpoint.

## LE — Learnable Examples / joint-conditional diffusion purification

- **Paper:** *Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples*, ACM Multimedia 2023. [Paper](https://arxiv.org/abs/2305.09241), [ACM record](https://doi.org/10.1145/3581783.3611833), [pinned source](https://github.com/jiangw-0/LE_JCDP/tree/52febef27e6657d3991da7263c38f3277a81e018).
- **Author entry points:** `main_Un.py`, `runners/empirical_unlearnable_c.py`, `purification/diff_purify.py`, and `configs/cifar10_Un.yml`.
- **Mechanism actually implemented:** conditional diffusion reverse purification steered jointly by pixel-level and perceptual similarity objectives. The repository example passes a classifier/diffusion class index as `model_kwargs["y"]`; the joint purification condition itself also uses image-based MSE/LPIPS guidance. These are distinct inputs and neither defines a multi-instance YOLO condition.
- **Parameters/checkpoints disclosed:** the checked CIFAR-10-Un example uses a CIFAR diffusion model, 100 configured epochs, batch 500, 10 attack steps/40 inner iterations, and purification settings including `max_iter=4`, `purify_step=20`, MSE/LPIPS guidance, and configured guidance scales. These are that repo's example values, not MI-UE Table 5 universal values or VOC defaults. The README lists CIFAR-10/100 model downloads, not a VOC20/640 detector model. No checkpoint was downloaded or loaded.
- **Data/threat-model need:** the method's diffusion model is trained on unprotected data. The exact auxiliary dataset, overlap, and access policy must be frozen for the intended experiment; clean VOC validation/test images or labels must never supply this training data. The current audit found no approved non-validation VOC-compatible model/data package in the project.
- **Detection gap:** one classifier label cannot encode all objects in a VOC frame; the repository's example label condition cannot be reused as a single VOC image class. A detection conditioning scheme and data-access policy would change the threat model and must be predeclared. Pixel/perceptual guidance also needs 640-resolution small-object and bbox-semantic validation.
- **S4 adaptation status:** **`BLOCKED_DEPENDENCY`** (with a separate scientific/threat-model adaptation still required). No compatible checkpoint or authorized auxiliary training-data source is available, and no detector adapter exists.
- **License:** no root license file was found at the pinned tree. No upstream code was copied.

## Fidelity conclusions

None of these four methods is a ready-made YOLOv8 purification call. OP and D-VAE require a defensible multi-instance algorithm definition. AVATAR requires an actual compatible pretrained model and resolution/semantic validation. LE additionally requires a locked auxiliary-data policy and detection conditioning. The only common implementation added in this branch checks sample/GT metadata invariants; it is not a defense implementation and cannot establish semantic fidelity.
