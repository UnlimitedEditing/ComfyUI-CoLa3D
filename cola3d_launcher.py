"""Run a CoLa3D script with Hugging Face downloads redirected to pre-staged local files.

Usage: python cola3d_launcher.py <script.py> [args...]
Env:   COLA3D_HF_MAP = JSON {"repo_id|filename": "/local/path"}
Every hf_hub_download for a mapped (repo, file) returns the local path instead of touching the
network, so the whole job stays inside the 180 s run budget (weights arrive via concept_mapping).
"""
import json
import os
import runpy
import sys

import huggingface_hub
import huggingface_hub.file_download as _fd

_MAP = json.loads(os.environ.get("COLA3D_HF_MAP", "{}"))
_orig = huggingface_hub.hf_hub_download


def _patched(repo_id=None, filename=None, *args, **kwargs):
    key = f"{repo_id}|{filename}"
    hit = _MAP.get(key)
    if hit:
        if not os.path.isfile(hit):
            raise FileNotFoundError(f"CoLa3D staged file missing for {key}: {hit}")
        print(f"[cola3d-launcher] {key} -> {hit}", flush=True)
        return hit
    print(f"[cola3d-launcher] UNMAPPED download {key} (will hit network)", flush=True)
    return _orig(repo_id, filename, *args, **kwargs)


huggingface_hub.hf_hub_download = _patched
_fd.hf_hub_download = _patched

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
