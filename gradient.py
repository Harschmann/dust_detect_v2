"""
gradient.py
------------
A lightweight animated "shader-like" gradient, drawn on a tk.Canvas.

It computes a low-resolution flowing colour field with numpy (a couple
of phase-shifted sinusoids blended between a few dark colours), upscales
it, and re-blits every frame via PIL. Cheap enough to run behind the UI
at ~20 fps. Used for the ambient background glow and the header, giving
the dark "stealth" interface a slow moving-gradient feel.
"""

import tkinter as tk
import numpy as np
from PIL import Image, ImageTk


class FlowGradient(tk.Canvas):
    def __init__(self, master, colors=None, speed=1.0, downscale=10, fps=20, **kw):
        super().__init__(master, highlightthickness=0, bd=0, **kw)
        # colours as RGB 0-255; kept dark for a stealth look
        self.colors = colors or [
            (10, 12, 20), (18, 24, 46), (28, 18, 52), (12, 30, 44), (8, 10, 16)
        ]
        self.speed = speed
        self.downscale = max(4, downscale)
        self.fps = fps
        self._t = 0.0
        self._photo = None
        self._img_id = None
        self._arr = np.array(self.colors, dtype=np.float32)
        self.bind("<Configure>", lambda e: None)
        self.after(60, self._tick)

    def _field(self, w, h, t):
        sw = max(2, w // self.downscale)
        sh = max(2, h // self.downscale)
        xs = np.linspace(0, 1, sw, dtype=np.float32)
        ys = np.linspace(0, 1, sh, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)

        # a few drifting waves combine into a smooth 0..1 selector field
        f = (np.sin(3.0 * gx + t) +
             np.sin(4.0 * gy - 0.7 * t) +
             np.sin(2.5 * (gx + gy) + 0.4 * t) +
             np.sin(5.0 * (gx - gy) - 0.3 * t))
        f = (f + 4.0) / 8.0                      # -> 0..1
        f = np.clip(f, 0, 1)

        # map the selector across the colour stops
        n = len(self.colors) - 1
        pos = f * n
        i0 = np.floor(pos).astype(np.int32)
        i0 = np.clip(i0, 0, n - 1)
        frac = (pos - i0)[..., None]
        c0 = self._arr[i0]
        c1 = self._arr[i0 + 1]
        rgb = (c0 * (1 - frac) + c1 * frac).astype(np.uint8)
        return rgb

    def _tick(self):
        w = self.winfo_width()
        h = self.winfo_height()
        if w > 2 and h > 2:
            small = self._field(w, h, self._t)
            img = Image.fromarray(small, "RGB").resize((w, h), Image.BILINEAR)
            self._photo = ImageTk.PhotoImage(img)
            if self._img_id is None:
                self._img_id = self.create_image(0, 0, anchor="nw", image=self._photo)
                self.tag_lower(self._img_id)
            else:
                self.itemconfig(self._img_id, image=self._photo)
                self.coords(self._img_id, 0, 0)
        self._t += 0.05 * self.speed
        self.after(int(1000 / self.fps), self._tick)

