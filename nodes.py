"""CoLa3D (Composable Latent 3D Decomposition) glue for ComfyUI / Graydient.

Wraps the official Master-cai/CoLa3D inference script (scripts/inference/run_inference.py) as a
subprocess. The repo itself is cloned into custom_nodes by the workflow's requirements.github, and
all weights are pre-staged under models/cola3d by concept_mapping, so nothing downloads at run time.
Output: one GLB holding every decomposed object mesh as its own node, in the shared scene frame.
"""
import glob
import io
import json
import os
import subprocess
import sys
import time

import numpy as np

try:
    import folder_paths
except Exception:  # allows importing outside ComfyUI for linting
    folder_paths = None

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "cola3d_launcher.py")


def _models_dir():
    base = folder_paths.models_dir if folder_paths else os.path.join(HERE, "models")
    return os.path.join(base, "cola3d")


def _custom_nodes_dir():
    if folder_paths and getattr(folder_paths, "base_path", None):
        return os.path.join(folder_paths.base_path, "custom_nodes")
    return os.path.dirname(HERE)


def _find_dir(marker, required=True):
    root = _custom_nodes_dir()
    for name in sorted(os.listdir(root)):
        cand = os.path.join(root, name)
        if os.path.isfile(os.path.join(cand, marker)):
            return cand
    if required:
        raise RuntimeError(
            f"Could not find a custom_nodes repo containing {marker} under {root}. "
            "The workflow must list Master-cai/CoLa3D and Pointcept/Utonia in requirements.github."
        )
    return None


def _fetch_image(value):
    """URL, local path, or saved filename -> PIL RGBA."""
    from PIL import Image

    value = (value or "").strip()
    if not value:
        raise RuntimeError("CoLa3D: empty image input (use `--init-image <url>`, not --init-image-url).")
    if value.startswith(("http://", "https://")):
        import requests

        r = requests.get(value, timeout=60, headers={"User-Agent": "cola3d-glue/1.0"})
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
    else:
        path = value
        if not os.path.isfile(path) and folder_paths:
            path = os.path.join(folder_paths.get_input_directory(), value)
        if not os.path.isfile(path):
            raise RuntimeError(f"CoLa3D: image not found: {value}")
        img = Image.open(path)
    return img.convert("RGBA")


class CoLa3DDecompose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "family": (["q8"], {"default": "q8"}),
                "max_side": ("INT", {"default": 0, "min": 0, "max": 2048}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("glb_path", "summary")
    FUNCTION = "run"
    CATEGORY = "CoLa3D"

    @staticmethod
    def _ensure_utils3d(models):
        """CoLa3D needs the pinned utils3d commit (has utils3d.torch). Use it if importable, else fetch the zip."""
        import importlib.util

        if importlib.util.find_spec("utils3d") is not None:
            return None
        import zipfile

        import requests

        dest = os.path.join(models, "utils3d_src")
        if not os.path.isdir(os.path.join(dest, "utils3d")):
            sha = "9a4eb15e4021b67b12c460c7057d642626897ec8"
            r = requests.get(f"https://github.com/EasternJournalist/utils3d/archive/{sha}.zip", timeout=120)
            r.raise_for_status()
            os.makedirs(dest, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(dest)
            inner = os.path.join(dest, f"utils3d-{sha}")
            if os.path.isdir(os.path.join(inner, "utils3d")):
                import shutil

                shutil.move(os.path.join(inner, "utils3d"), os.path.join(dest, "utils3d"))
        print(f"[CoLa3D] utils3d not pip-installed; using pinned source at {dest}", flush=True)
        return dest

    # module -> wheel/pip spec; installed here only if the workflow's requirements.pip did not deliver them
    _REL = "https://github.com/UnlimitedEditing/{r}/releases/download/v1.0%2Bpt2110cu128/{w}"
    _DEPS = [
        ("trimesh", "trimesh"), ("plyfile", "plyfile"), ("zstandard", "zstandard"), ("rtree", "rtree"),
        ("addict", "addict"), ("skimage", "scikit-image"), ("sklearn", "scikit-learn"), ("moderngl", "moderngl"),
        ("imageio", "imageio"), ("matplotlib", "matplotlib"), ("lpips", "lpips"), ("timm", "timm"),
        ("flash_attn", "https://github.com/UnlimitedEditing/flash-attention/releases/download/v2.8.4%2Bcu128torch2.11/flash_attn-2.8.4%2Bcu128torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"),
        ("nvdiffrast", _REL.format(r="nvdiffrast", w="nvdiffrast-0.4.0-cp312-cp312-linux_x86_64.whl")),
        ("cumesh", _REL.format(r="CuMesh", w="cumesh-1.0-cp312-cp312-linux_x86_64.whl")),
        ("flex_gemm", _REL.format(r="FlexGEMM", w="flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl")),
        ("o_voxel", _REL.format(r="TRELLIS.2", w="o_voxel-0.0.1-cp312-cp312-linux_x86_64.whl")),
        ("spconv", "spconv-cu126"),
        ("easydict", "easydict"), ("commentjson", "commentjson"), ("pandas", "pandas"),
        ("cv2", "opencv-python-headless"), ("open3d", "open3d"),
    ]

    @classmethod
    def _ensure_deps(cls):
        for mod, spec in cls._DEPS:
            probe = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True, text=True)
            if probe.returncode == 0:
                continue
            print(f"[CoLa3D] import {mod} fails ({probe.stderr.strip()[-200:]}): pip install {spec}", flush=True)
            args = [sys.executable, "-m", "pip", "install", "--no-compile", spec]
            if spec.startswith("http"):
                args.insert(4, "--no-deps")
            out = subprocess.run(args, capture_output=True, text=True)
            print(f"[CoLa3D] pip {mod} rc={out.returncode}: {(out.stdout + out.stderr)[-800:]}", flush=True)
            probe = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True, text=True)
            print(f"[CoLa3D] after install import {mod} rc={probe.returncode} {probe.stderr.strip()[-300:]}", flush=True)

    def run(self, image_url, seed, family, max_side):
        self._ensure_deps()
        t0 = time.time()
        repo = _find_dir(os.path.join("scripts", "inference", "run_inference.py"))
        utonia = _find_dir(os.path.join("utonia", "model.py"), required=False)  # None if pip-installed
        models = _models_dir()
        ckpt_dir = os.path.join(models, "checkpoints")

        img = _fetch_image(image_url)
        arr = np.asarray(img)
        if arr[..., :3].std() < 2.0:
            raise RuntimeError("CoLa3D: input image is blank (std<2) - the image never arrived.")
        if max_side and max(img.size) > max_side:
            s = max_side / max(img.size)
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))))

        work = os.path.join(models, "work", f"run_{int(time.time())}")
        os.makedirs(work, exist_ok=True)
        in_png = os.path.join(work, "scene.png")
        img.save(in_png)
        out_dir = os.path.join(work, "out")

        # --- stage pretrained_ckpts for the repo (the script reads it relative to cwd = repo root) ---
        base = os.path.join(models, "trellis2_4b")
        pre = os.path.join(repo, "pretrained_ckpts")
        os.makedirs(os.path.join(pre, "ckpts"), exist_ok=True)
        with open(os.path.join(base, "pipeline.json")) as f:
            pipe = json.load(f)
        pipe["args"]["image_cond_model"]["args"]["model_name"] = os.path.join(models, "dinov3")
        with open(os.path.join(pre, "pipeline.json"), "w") as f:
            json.dump(pipe, f, indent=2)
        for fn in glob.glob(os.path.join(base, "ckpts", "*")):
            dst = os.path.join(pre, "ckpts", os.path.basename(fn))
            if not os.path.exists(dst):
                try:
                    os.symlink(fn, dst)
                except OSError:
                    import shutil

                    shutil.copy2(fn, dst)

        with open(os.path.join(repo, "configs", "inference", f"{family}.json")) as f:
            config = json.load(f)
        config["settings"]["seed"] = int(seed)
        ck = config["checkpoints"]
        for key in ("free", "geometry_decoder", "ss_flow", "slat_flow"):
            p = os.path.join(ckpt_dir, ck[key])
            if not os.path.isfile(p):
                raise RuntimeError(f"CoLa3D: missing staged checkpoint {p} (concept_mapping did not deliver it).")

        cmd = [
            sys.executable, LAUNCHER, os.path.join(repo, "scripts", "inference", "run_inference.py"),
            "--dataset-class", "single_image", "--output-dir", out_dir,
            "--decoder-config", os.path.join(repo, config["model_configs"]["free"]),
            "--ss-flow-config", os.path.join(repo, config["model_configs"]["ss_flow"]),
            "--pipeline-config", os.path.join(pre, "pipeline.json"),
        ]
        for flag, key in (("cola3d-ckpt", "free"), ("geo-ckpt", "geometry_decoder"),
                          ("ss-flow-ckpt", "ss_flow"), ("ss-flow-ema-ckpt", "ss_flow"),
                          ("slat-flow-ckpt", "slat_flow")):
            cmd += ["--" + flag, os.path.join(ckpt_dir, ck[key])]
        cmd += ["--input-rgba", in_png]
        for key, value in config["settings"].items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd += [flag, str(value)]

        hf_map = {
            "microsoft/TRELLIS-image-large|ckpts/ss_dec_conv3d_16l8_fp16.json":
                os.path.join(models, "trellis_image_large", "ss_dec_conv3d_16l8_fp16.json"),
            "microsoft/TRELLIS-image-large|ckpts/ss_dec_conv3d_16l8_fp16.safetensors":
                os.path.join(models, "trellis_image_large", "ss_dec_conv3d_16l8_fp16.safetensors"),
            "Pointcept/Utonia|utonia.pth": os.path.join(models, "utonia", "utonia.pth"),
        }
        utils3d_dir = self._ensure_utils3d(models)
        env = dict(os.environ)
        pp = [repo] + ([utonia] if utonia else []) + ([utils3d_dir] if utils3d_dir else []) + [env.get("PYTHONPATH", "")]
        env["PYTHONPATH"] = os.pathsep.join(p for p in pp if p)
        env["COLA3D_HF_MAP"] = json.dumps(hf_map)
        env["HF_HUB_OFFLINE"] = "1"
        env.setdefault("PYTHONUNBUFFERED", "1")

        diag_mods = ('cumesh', 'flex_gemm', 'o_voxel', 'nvdiffrast', 'flash_attn', 'spconv', 'trimesh')
        diag_code = 'import sys; print(sys.path); ' + '; '.join(
            f"exec('try: import {m}; print(\"OK {m}\")'+chr(10)+'except Exception as e: print(\"FAIL {m}\", repr(e))')" for m in diag_mods)
        dg = subprocess.run([sys.executable, '-c', diag_code], cwd=repo, env=env, capture_output=True, text=True)
        print('[CoLa3D] diag: ' + dg.stdout + dg.stderr[-500:], flush=True)
        pl = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
        print("[CoLa3D] pip list (filtered): " + " | ".join(l for l in pl.stdout.splitlines()
              if any(k in l.lower() for k in ("cumesh", "flex", "voxel", "nvdiff", "flash", "spconv", "cumm", "torch", "utils3d", "open3d"))), flush=True)
        print(f"[CoLa3D] running: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        tail = []
        for line in proc.stdout:
            print("[CoLa3D] " + line.rstrip(), flush=True)
            tail.append(line)
            tail = tail[-60:]
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"CoLa3D inference failed (exit {rc}). Last output:\n" + "".join(tail[-40:]))

        pat = os.path.join(out_dir, "**", "geometry", "post_standard")
        parts = sorted(glob.glob(os.path.join(pat, "part_*.glb"), recursive=True))
        full = glob.glob(os.path.join(pat, "full_scene.glb"), recursive=True)
        if not parts and not full:
            raise RuntimeError("CoLa3D finished but produced no part_*.glb / full_scene.glb.\n" + "".join(tail[-30:]))

        import trimesh

        if parts:
            scene = trimesh.Scene()
            for i, p in enumerate(parts):
                loaded = trimesh.load(p, force="scene")
                for name, geom in loaded.geometry.items():
                    scene.add_geometry(geom, node_name=f"object_{i}", geom_name=f"object_{i}_{name}")
        else:
            scene = trimesh.load(full[0], force="scene")
        glb_path = os.path.join(work, "cola3d_objects.glb")
        scene.export(glb_path)
        summary = f"{len(parts)} object meshes; {os.path.getsize(glb_path)/1e6:.1f} MB GLB; total {time.time()-t0:.1f}s"
        print("[CoLa3D] " + summary, flush=True)
        return (glb_path, summary)


NODE_CLASS_MAPPINGS = {"CoLa3DDecompose": CoLa3DDecompose}
NODE_DISPLAY_NAME_MAPPINGS = {"CoLa3DDecompose": "CoLa3D Decompose (Free)"}
