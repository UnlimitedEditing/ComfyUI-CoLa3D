# ComfyUI-CoLa3D

Glue node for [CoLa3D](https://github.com/Master-cai/CoLa3D) (Composable Latent 3D Decomposition, NeurIPS 2026):
one scene image in, a single GLB out holding every decomposed object mesh in a shared coordinate frame.

`CoLa3DDecompose` runs the official `scripts/inference/run_inference.py` (Free mode, q8 family) as a subprocess.
It expects the CoLa3D and Utonia repos cloned into `custom_nodes`, weights staged under `models/cola3d/`
(checkpoints, trellis2_4b, trellis_image_large, dinov3, utonia) and the compiled deps (flash-attn, FlexGEMM, CuMesh,
O-Voxel, nvdiffrast, spconv) installed. See `gen_cola3d.py` in the Graydient workflow tooling for the exact recipe.
Hugging Face downloads are redirected to the staged files by `cola3d_launcher.py`.
