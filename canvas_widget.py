"""
canvas_widget.py
------------------
The one image-viewer widget used everywhere in the app (live feed,
and the original/after review panels). Same controls everywhere:

    Mouse wheel                -> zoom, anchored at the cursor
    Click + drag                -> pan
    Plain click (<4px move)     -> on_click callback (add/select ROI)
    Shift + wheel (editable)    -> live radius adjust on selected ROI
    [ / ]  or  Up / Down         -> live radius adjust (keyboard)
    Delete / Backspace          -> remove selected ROI

Two canvases can share one ViewState instance to stay perfectly in
sync - used for the original-vs-after side-by-side review.
"""

import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk


class ViewState:
    def __init__(self, min_zoom=0.05, max_zoom=12.0):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self._subs = []

    def subscribe(self, redraw_callback):
        self._subs.append(redraw_callback)

    def _notify(self):
        for cb in self._subs:
            cb()

    def zoom_at(self, cx, cy, factor):
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))
        if new_zoom == self.zoom:
            return
        nx = (cx - self.pan_x) / self.zoom
        ny = (cy - self.pan_y) / self.zoom
        self.pan_x = cx - nx * new_zoom
        self.pan_y = cy - ny * new_zoom
        self.zoom = new_zoom
        self._notify()

    def pan_by(self, dx, dy):
        self.pan_x += dx
        self.pan_y += dy
        self._notify()

    def fit(self, img_w, img_h, canvas_w, canvas_h):
        if img_w <= 0 or img_h <= 0 or canvas_w <= 1 or canvas_h <= 1:
            return
        scale = min(canvas_w / img_w, canvas_h / img_h) * 0.96
        self.zoom = scale
        self.pan_x = (canvas_w - img_w * scale) / 2
        self.pan_y = (canvas_h - img_h * scale) / 2
        self._notify()

    def to_canvas(self, x, y):
        return x * self.zoom + self.pan_x, y * self.zoom + self.pan_y

    def to_native(self, x, y):
        return (x - self.pan_x) / self.zoom, (y - self.pan_y) / self.zoom


class ImageCanvas(tk.Canvas):
    def __init__(self, master, view_state=None, editable=False,
                 on_click=None, on_radius_change=None, on_delete=None,
                 bg="#1a1d21", **kwargs):
        super().__init__(master, bg=bg, highlightthickness=0, **kwargs)
        self.view = view_state or ViewState()
        self.view.subscribe(self.redraw)
        self.editable = editable
        self.on_click = on_click
        self.on_radius_change = on_radius_change
        self.on_delete = on_delete

        self.source_image = None  # native-resolution BGR numpy array
        self._tk_img = None
        self._press = None
        self._dragged = False

        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", lambda e: self._on_wheel(e, delta=120))
        self.bind("<Button-5>", lambda e: self._on_wheel(e, delta=-120))
        self.bind("<Shift-MouseWheel>", self._on_shift_wheel)
        self.bind("<Shift-Button-4>", lambda e: self._on_shift_wheel(e, delta=120))
        self.bind("<Shift-Button-5>", lambda e: self._on_shift_wheel(e, delta=-120))
        self.bind("<Key-Delete>", lambda e: self.on_delete() if self.on_delete else None)
        self.bind("<Key-BackSpace>", lambda e: self.on_delete() if self.on_delete else None)
        self.bind("<Key-bracketright>", lambda e: self._nudge_radius(4))
        self.bind("<Key-bracketleft>", lambda e: self._nudge_radius(-4))
        self.bind("<Up>", lambda e: self._nudge_radius(4))
        self.bind("<Down>", lambda e: self._nudge_radius(-4))
        self.bind("<Enter>", lambda e: self.focus_set())

    # ------------------------------------------------------------ input
    def _on_press(self, e):
        self._press = (e.x, e.y)
        self._dragged = False
        self.focus_set()

    def _on_drag(self, e):
        if self._press is None:
            return
        dx, dy = e.x - self._press[0], e.y - self._press[1]
        if abs(dx) > 4 or abs(dy) > 4:
            self._dragged = True
        self.view.pan_by(dx, dy)
        self._press = (e.x, e.y)

    def _on_release(self, e):
        if self._press is not None and not self._dragged and self.editable and self.on_click:
            nx, ny = self.view.to_native(e.x, e.y)
            self.on_click(nx, ny, e.x, e.y)
        self._press = None
        self._dragged = False

    def _on_wheel(self, e, delta=None):
        d = delta if delta is not None else e.delta
        factor = 1.1 if d > 0 else (1 / 1.1)
        self.view.zoom_at(e.x, e.y, factor)

    def _on_shift_wheel(self, e, delta=None):
        d = delta if delta is not None else e.delta
        self._nudge_radius(6 if d > 0 else -6)

    def _nudge_radius(self, delta_px):
        if self.editable and self.on_radius_change:
            self.on_radius_change(delta_px)

    # ------------------------------------------------------------- draw
    def set_image(self, bgr_image, fit_if_first=True):
        first = self.source_image is None
        self.source_image = bgr_image
        if first and fit_if_first and bgr_image is not None:
            h, w = bgr_image.shape[:2]
            self._try_auto_fit(w, h, 0)
        self.redraw()

    def _try_auto_fit(self, w, h, attempts):
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw > 1 and ch > 1:
            self.view.fit(w, h, cw, ch)
        elif attempts < 25:
            self.after(30, lambda: self._try_auto_fit(w, h, attempts + 1))

    def fit_to_window(self):
        if self.source_image is None:
            return
        h, w = self.source_image.shape[:2]
        self.view.fit(w, h, max(self.winfo_width(), 1), max(self.winfo_height(), 1))

    def redraw(self):
        self.delete("all")
        if self.source_image is None:
            return
        cw, ch = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        z = self.view.zoom
        m = np.array([[z, 0, self.view.pan_x],
                      [0, z, self.view.pan_y]], dtype=np.float32)
        canvas_img = cv2.warpAffine(self.source_image, m, (cw, ch),
                                     flags=cv2.INTER_LINEAR, borderValue=(26, 29, 33))
        rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.create_image(0, 0, anchor="nw", image=self._tk_img)

