"""
storage.py
-----------
Per-model config (ROIs + detection params) and capture folders:

    inspection_data/<model>/original/NG|OK/<model>_test<N>_<NG|OK>.png
    inspection_data/<model>/after/NG|OK/<model>_test<N>_<NG|OK>.png
    configs/<model>.json

"original" is the raw frame, "after" has the defect circles drawn.
<N> is one running counter per model (NG and OK share it). It's derived
by scanning existing filenames for the max index, so deleting or
overriding a capture never causes a filename collision.
"""

import json
import os
import re
import cv2


class Storage:
    def __init__(self, base_dir="inspection_data", config_dir="configs"):
        self.base_dir = base_dir
        self.config_dir = config_dir
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)

    # ---------------------------------------------------------- config
    def _cfg_path(self, model):
        return os.path.join(self.config_dir, f"{model}.json")

    def list_models(self):
        return sorted(f[:-5] for f in os.listdir(self.config_dir) if f.endswith(".json"))

    def save_config(self, model, rois, params):
        with open(self._cfg_path(model), "w") as f:
            json.dump({"model": model, "rois": rois, "params": params}, f, indent=2)

    def load_config(self, model):
        path = self._cfg_path(model)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def delete_config(self, model):
        p = self._cfg_path(model)
        if os.path.exists(p):
            os.remove(p)

    # --------------------------------------------------------- captures
    def _dirs(self, model):
        d = {}
        for stage in ("original", "after"):
            for verdict in ("NG", "OK"):
                path = os.path.join(self.base_dir, model, stage, verdict)
                os.makedirs(path, exist_ok=True)
                d[(stage, verdict)] = path
        return d

    def next_index(self, model):
        dirs = self._dirs(model)
        pat = re.compile(rf"^{re.escape(model)}_test(\d+)_(NG|OK)\.png$")
        mx = 0
        for verdict in ("NG", "OK"):
            for f in os.listdir(dirs[("original", verdict)]):
                m = pat.match(f)
                if m:
                    mx = max(mx, int(m.group(1)))
        return mx + 1

    def save_capture(self, model, original, annotated, verdict):
        dirs = self._dirs(model)
        idx = self.next_index(model)
        name = f"{model}_test{idx}_{verdict}.png"
        orig = os.path.join(dirs[("original", verdict)], name)
        after = os.path.join(dirs[("after", verdict)], name)
        cv2.imwrite(orig, original)
        cv2.imwrite(after, annotated)
        return {"index": idx, "verdict": verdict, "original_path": orig, "after_path": after}

    def relabel(self, model, index, old, new):
        if old == new:
            return
        dirs = self._dirs(model)
        old_name = f"{model}_test{index}_{old}.png"
        new_name = f"{model}_test{index}_{new}.png"
        for stage in ("original", "after"):
            src = os.path.join(dirs[(stage, old)], old_name)
            if os.path.exists(src):
                os.replace(src, os.path.join(dirs[(stage, new)], new_name))

    def stats(self, model):
        dirs = self._dirs(model)
        def count(v):
            return len([f for f in os.listdir(dirs[("original", v)]) if f.lower().endswith(".png")])
        ng, ok = count("NG"), count("OK")
        total = ng + ok
        return {"ng": ng, "ok": ok, "total": total, "ng_rate": (ng / total * 100) if total else 0.0}
