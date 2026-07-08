"""
main.py
--------
Camera-module dust & string inspector - lightweight build.

Run:
    python main.py

Pick a camera (Basler / any webcam / synthetic) -> click the live view
to mark each lens centre -> tune radius -> save under a phone-model name
-> Capture & Inspect runs the simple Z-score white-dust/string detector
and files the result as NG/OK.

Performance notes:
  * The live view is shown DOWNSCALED (big industrial frames are 20MP;
    pushing every full-res frame through the UI 30x/sec is what made it
    hang). ROIs are still kept in full-resolution coordinates and the
    actual inspection always runs on the full-res frame - only the
    on-screen preview is scaled down.
  * No animated backgrounds - plain dark UI, minimal redraw work.
"""

import cv2
import customtkinter as ctk
from tkinter import messagebox
from datetime import datetime
import threading

from camera import CameraManager, enumerate_cameras
from detector import inspect, DEFAULT_PARAMS
from storage import Storage
from canvas_widget import ViewState, ImageCanvas

ctk.set_appearance_mode("dark")

# minimal dark palette
BG = "#0e0f12"
PANEL = "#16181d"
PANEL2 = "#101216"
ACCENT = "#4fd0e0"
NG = "#ff5468"
OK = "#37e29a"
MUTED = "#6b7280"
TEXT = "#e6e8ec"

ROI_COLOR = (150, 150, 158)     # BGR unselected
ROI_SEL = (0, 210, 255)         # BGR selected

MAX_PREVIEW = 1500              # longest side (px) of the on-screen live preview


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Dust Inspector")
        self.geometry("1360x860")
        self.minsize(1100, 700)
        self.configure(fg_color=BG)

        self.storage = Storage()
        self.camera = CameraManager()
        self.cameras = []
        self._cam_labels = {}
        self._pending_cams = None

        self.rois = []                 # full-resolution coords: {cx,cy,r}
        self.selected_idx = None
        self.roi_rows = []
        self._param_widgets = {}
        self._disp_scale = 1.0         # preview downscale factor (<=1)
        self._frame_hw = (0, 0)        # full-res frame size
        self._res_autoset_done = False
        self._last_result = None

        self.current_model = ctk.StringVar(value="")
        self.sigma = ctk.DoubleVar(value=DEFAULT_PARAMS["sigma"])
        self.min_area = ctk.IntVar(value=DEFAULT_PARAMS["min_area"])
        self.white_only = ctk.BooleanVar(value=DEFAULT_PARAMS["white_only"])
        self.white_sat = ctk.IntVar(value=DEFAULT_PARAMS["white_max_saturation"])

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<space>", self._on_space)

        threading.Thread(target=self._startup_cameras, daemon=True).start()
        self._refresh_models()
        self._tick()

    # ---------------------------------------------------------- params
    def params(self):
        return {
            "sigma": float(self.sigma.get()),
            "min_area": int(self.min_area.get()),
            "white_only": bool(self.white_only.get()),
            "white_max_saturation": int(self.white_sat.get()),
        }

    def _apply_params(self, p):
        self.sigma.set(p.get("sigma", DEFAULT_PARAMS["sigma"]))
        self.min_area.set(int(p.get("min_area", DEFAULT_PARAMS["min_area"])))
        self.white_only.set(bool(p.get("white_only", DEFAULT_PARAMS["white_only"])))
        self.white_sat.set(int(p.get("white_max_saturation", DEFAULT_PARAMS["white_max_saturation"])))
        for lbl, fmt, var in self._param_widgets.values():
            lbl.configure(text=fmt(var.get()))

    # =============================================================== UI
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # --- top bar (static, minimal)
        top = ctk.CTkFrame(self, height=54, corner_radius=0, fg_color=PANEL)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        top.grid_propagate(False)
        ctk.CTkLabel(top, text="  \u25C9  Dust Inspector",
                     font=ctk.CTkFont(size=17, weight="bold"), text_color=TEXT
                     ).pack(side="left", padx=12, pady=10)
        self.status = ctk.CTkLabel(top, text="", text_color=MUTED, font=ctk.CTkFont(size=12))
        self.status.pack(side="left", padx=8)

        cam_box = ctk.CTkFrame(top, fg_color="transparent")
        cam_box.pack(side="right", padx=10)
        ctk.CTkLabel(cam_box, text="Camera", text_color=MUTED,
                     font=ctk.CTkFont(size=11)).pack(side="left", padx=(0, 6))
        self.cam_combo = ctk.CTkComboBox(cam_box, width=290, values=["Scanning..."],
                                         command=self._on_camera_pick, fg_color=PANEL2,
                                         button_color="#2b8f9c", border_color="#242a34",
                                         dropdown_fg_color=PANEL)
        self.cam_combo.pack(side="left", padx=4)
        ctk.CTkButton(cam_box, text="\u21BB", width=36, fg_color=PANEL2, hover_color="#1b1f27",
                      command=self._rescan_cameras).pack(side="left")

        # --- live view (left)
        view = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        view.grid(row=1, column=0, sticky="nsew", padx=(10, 6), pady=(8, 0))
        view.grid_rowconfigure(0, weight=1)
        view.grid_columnconfigure(0, weight=1)
        self.view = ViewState()
        self.canvas = ImageCanvas(view, view_state=self.view, editable=True, bg="#050608",
                                  on_click=self._on_click, on_radius_change=self._nudge_radius,
                                  on_delete=self._delete_selected)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        bar = ctk.CTkFrame(view, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ctk.CTkButton(bar, text="\u2212", width=34, fg_color=PANEL, hover_color="#1b1f27",
                      command=lambda: self._zoom(1 / 1.2)).pack(side="left", padx=3)
        ctk.CTkButton(bar, text="\u2922 Fit", width=58, fg_color=PANEL, hover_color="#1b1f27",
                      command=self.canvas.fit_to_window).pack(side="left", padx=3)
        ctk.CTkButton(bar, text="+", width=34, fg_color=PANEL, hover_color="#1b1f27",
                      command=lambda: self._zoom(1.2)).pack(side="left", padx=3)
        ctk.CTkLabel(bar, text="click = ROI  \u2022  drag = pan  \u2022  scroll = zoom  \u2022  "
                              "shift+scroll = radius  \u2022  del = remove  \u2022  space = inspect",
                     text_color=MUTED, font=ctk.CTkFont(size=11)).pack(side="left", padx=12)

        # --- sidebar (right)
        side = ctk.CTkScrollableFrame(self, width=330, fg_color=PANEL, corner_radius=0,
                                      scrollbar_button_color="#2a2f3a")
        side.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        self._build_sidebar(side)

    def _section(self, parent, title):
        ctk.CTkLabel(parent, text=title.upper(), font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=ACCENT).pack(anchor="w", padx=8, pady=(16, 6))

    def _slider(self, parent, key, title, var, lo, hi, fmt, steps=None):
        ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(size=11), text_color=TEXT
                     ).pack(anchor="w", padx=8, pady=(8, 0))
        val = ctk.CTkLabel(parent, text=fmt(var.get()), text_color=MUTED, font=ctk.CTkFont(size=11))
        kw = {"number_of_steps": steps} if steps else {}
        ctk.CTkSlider(parent, from_=lo, to=hi, variable=var, progress_color=ACCENT,
                      button_color=ACCENT, button_hover_color=TEXT,
                      command=lambda v, l=val, f=fmt: l.configure(text=f(float(v))), **kw
                      ).pack(fill="x", padx=8)
        val.pack(anchor="e", padx=8)
        self._param_widgets[key] = (val, fmt, var)

    def _build_sidebar(self, p):
        self._section(p, "Phone model")
        self.model_combo = ctk.CTkComboBox(p, values=[], variable=self.current_model,
                                           command=lambda _c: self._load_model(), fg_color=PANEL2,
                                           button_color="#2b8f9c", border_color="#242a34",
                                           dropdown_fg_color=PANEL)
        self.model_combo.pack(fill="x", padx=8)
        row = ctk.CTkFrame(p, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=6)
        ctk.CTkButton(row, text="Load", fg_color=PANEL2, hover_color="#1b1f27",
                      command=self._load_model).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ctk.CTkButton(row, text="Save", fg_color=ACCENT, text_color="#04252b", hover_color=TEXT,
                      command=self._save_model).pack(side="left", expand=True, fill="x", padx=3)
        ctk.CTkButton(row, text="\U0001F5D1", width=38, fg_color="#4a1d24", hover_color="#6a2833",
                      command=self._delete_model).pack(side="left", padx=(3, 0))
        self.save_note = ctk.CTkLabel(p, text="", text_color=OK, font=ctk.CTkFont(size=11))
        self.save_note.pack(anchor="w", padx=8)

        self._section(p, "ROI points")
        self.roi_frame = ctk.CTkFrame(p, fg_color="transparent")
        self.roi_frame.pack(fill="x", padx=8)
        self.radius = ctk.CTkSlider(p, from_=10, to=1400, progress_color=ACCENT,
                                    button_color=ACCENT, button_hover_color=TEXT,
                                    command=self._radius_slider)
        self.radius.set(120)
        self.radius.pack(fill="x", padx=8, pady=(10, 0))
        self.radius_note = ctk.CTkLabel(p, text="select an ROI to edit its radius",
                                        text_color=MUTED, font=ctk.CTkFont(size=11))
        self.radius_note.pack(anchor="w", padx=8)
        self._rebuild_roi_list()

        self._section(p, "Detection")
        self._slider(p, "sigma", "Sensitivity (Z-score)  \u2013 lower = stricter",
                     self.sigma, 2.0, 6.0, lambda v: f"{v:.1f}")
        self._slider(p, "min_area", "Min defect size (pixels)",
                     self.min_area, 1, 2000, lambda v: str(int(v)))
        self._slider(p, "white_sat", "White strictness  \u2013 lower = only pure white",
                     self.white_sat, 20, 180, lambda v: str(int(v)))
        sw = ctk.CTkFrame(p, fg_color="transparent")
        sw.pack(fill="x", padx=8, pady=(8, 0))
        ctk.CTkSwitch(sw, text="White defects only (ignore colour)", variable=self.white_only,
                      progress_color=ACCENT).pack(anchor="w")

        self._section(p, "Inspect")
        ctk.CTkButton(p, text="\U0001F50D   CAPTURE & INSPECT", height=46, fg_color=ACCENT,
                      text_color="#04252b", hover_color=TEXT,
                      font=ctk.CTkFont(size=14, weight="bold"),
                      command=self._capture).pack(fill="x", padx=8, pady=4)

        self._section(p, "Stats")
        self.stats = ctk.CTkLabel(p, text="no captures yet", justify="left", anchor="w",
                                  text_color=TEXT, font=ctk.CTkFont(size=12))
        self.stats.pack(fill="x", padx=8)

        self._section(p, "Recent")
        self.log = ctk.CTkFrame(p, fg_color="transparent")
        self.log.pack(fill="x", padx=8, pady=(0, 16))
        self.log_rows = []

    # ============================================================ cameras
    def _startup_cameras(self):
        self._pending_cams = self.camera.connect_best_available()

    def _rescan_cameras(self):
        self.cam_combo.configure(values=["Scanning..."])
        self.cam_combo.set("Scanning...")
        threading.Thread(target=lambda: setattr(self, "_pending_cams", enumerate_cameras()),
                         daemon=True).start()

    def _populate_cameras(self, cams):
        self.cameras = cams
        self._cam_labels = {c["name"]: c for c in cams}
        self.cam_combo.configure(values=[c["name"] for c in cams] + ["Synthetic feed (no camera)"])
        self.cam_combo.set(self.camera.active["name"] if self.camera.active
                           else "Synthetic feed (no camera)")

    def _on_camera_pick(self, label):
        desc = self._cam_labels.get(label)
        self.cam_combo.set(label if label else "Synthetic feed (no camera)")
        threading.Thread(target=lambda: self.camera.connect(desc), daemon=True).start()

    # ============================================================= models
    def _refresh_models(self):
        self.model_combo.configure(values=self.storage.list_models())

    def _load_model(self):
        model = self.current_model.get().strip()
        if not model:
            return
        cfg = self.storage.load_config(model)
        self.rois = cfg.get("rois", []) if cfg else []
        if cfg:
            self._apply_params(cfg.get("params", {}))
        self.selected_idx = None
        self._rebuild_roi_list()
        self._update_stats()

    def _save_model(self):
        model = self.current_model.get().strip()
        if not model:
            messagebox.showwarning("Model name needed", "Pehle phone model ka naam likho ya select karo.")
            return
        self.storage.save_config(model, self.rois, self.params())
        self._refresh_models()
        self.current_model.set(model)
        self._update_stats()
        self.save_note.configure(text=f"\u2713 saved '{model}' - {len(self.rois)} ROI(s)")
        self.after(2200, lambda: self.save_note.configure(text=""))

    def _delete_model(self):
        model = self.current_model.get().strip()
        if not model:
            return
        if not messagebox.askyesno("Delete config", f"Delete saved config for '{model}'?\n"
                                                      "(captured photos stay.)"):
            return
        self.storage.delete_config(model)
        self._refresh_models()
        self.current_model.set("")
        self.rois, self.selected_idx = [], None
        self._rebuild_roi_list()

    # ============================================================ ROI edit
    # NOTE: canvas coords are in DISPLAY (downscaled) space; ROIs are stored
    # in FULL-RES space. Convert at the boundary with self._disp_scale.
    def _on_click(self, nx, ny, cx, cy):
        hit = self._hit(cx, cy)
        if hit is not None:
            self.selected_idx = hit
        else:
            fx, fy = nx / self._disp_scale, ny / self._disp_scale
            base = min(self._frame_hw) if min(self._frame_hw) > 0 else 960
            self.rois.append({"cx": fx, "cy": fy, "r": max(20, int(base * 0.04))})
            self.selected_idx = len(self.rois) - 1
        self._rebuild_roi_list()

    def _hit(self, cx, cy, tol=14):
        best, bd = None, tol
        for i, r in enumerate(self.rois):
            rx, ry = self.view.to_canvas(r["cx"] * self._disp_scale, r["cy"] * self._disp_scale)
            d = ((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5
            if d < bd:
                best, bd = i, d
        return best

    def _nudge_radius(self, delta):
        if self.selected_idx is None or self.selected_idx >= len(self.rois):
            return
        # delta is in display px; convert to full-res
        self.rois[self.selected_idx]["r"] = max(4, self.rois[self.selected_idx]["r"] + delta / self._disp_scale)
        self._rebuild_roi_list()

    def _radius_slider(self, value):
        if self.selected_idx is None or self.selected_idx >= len(self.rois):
            return
        self.rois[self.selected_idx]["r"] = float(value)
        self.radius_note.configure(text=f"ROI #{self.selected_idx + 1}  radius {int(float(value))}px")
        if self.selected_idx < len(self.roi_rows):
            self.roi_rows[self.selected_idx].configure(
                text=f"#{self.selected_idx + 1}    r = {int(float(value))}px")

    def _delete_selected(self):
        if self.selected_idx is None or self.selected_idx >= len(self.rois):
            return
        del self.rois[self.selected_idx]
        self.selected_idx = None
        self._rebuild_roi_list()

    def _select(self, i):
        self.selected_idx = i
        self._rebuild_roi_list()

    def _delete_at(self, i):
        del self.rois[i]
        if self.selected_idx == i:
            self.selected_idx = None
        elif self.selected_idx is not None and self.selected_idx > i:
            self.selected_idx -= 1
        self._rebuild_roi_list()

    def _rebuild_roi_list(self):
        for w in self.roi_frame.winfo_children():
            w.destroy()
        self.roi_rows = []
        if not self.rois:
            ctk.CTkLabel(self.roi_frame, text="click the live view to add one",
                         text_color=MUTED, font=ctk.CTkFont(size=11)).pack(anchor="w", pady=2)
        for i, r in enumerate(self.rois):
            sel = i == self.selected_idx
            rowf = ctk.CTkFrame(self.roi_frame, fg_color=("#132a30" if sel else "transparent"))
            rowf.pack(fill="x", pady=1)
            btn = ctk.CTkButton(rowf, text=f"#{i + 1}    r = {int(r['r'])}px", anchor="w",
                                fg_color="transparent", text_color=(ACCENT if sel else TEXT),
                                hover_color="#1b1f27", command=lambda idx=i: self._select(idx))
            btn.pack(side="left", expand=True, fill="x")
            self.roi_rows.append(btn)
            ctk.CTkButton(rowf, text="\u2715", width=28, fg_color="transparent",
                          hover_color="#6a2833", command=lambda idx=i: self._delete_at(idx)
                          ).pack(side="right")
        if self.selected_idx is not None and self.selected_idx < len(self.rois):
            self.radius.set(min(self.rois[self.selected_idx]["r"], 1400))
            self.radius_note.configure(
                text=f"ROI #{self.selected_idx + 1}  radius {int(self.rois[self.selected_idx]['r'])}px")
        else:
            self.radius_note.configure(text="select an ROI to edit its radius")

    # ================================================================ zoom
    def _zoom(self, f):
        self.view.zoom_at(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, f)

    # ============================================================= inspect
    def _on_space(self, _e):
        w = self.focus_get()
        if w is not None and w.winfo_class() in ("Entry", "TEntry", "TCombobox", "Text", "Spinbox"):
            return
        self._capture()

    def _capture(self):
        model = self.current_model.get().strip()
        if not model:
            messagebox.showwarning("Model name needed", "Pehle model select ya type karo.")
            return
        if not self.rois:
            messagebox.showwarning("No ROI", "Kam se kam ek ROI lagao (live view pe click karke).")
            return
        frame = self.camera.get_frame()          # FULL resolution
        if frame is None:
            messagebox.showwarning("No frame", "Camera se frame nahi mila.")
            return
        try:
            res = inspect(frame, self.rois, self.params())
            saved = self.storage.save_capture(model, frame, res.annotated, res.verdict)
        except Exception as e:
            messagebox.showerror("Inspection failed", str(e))
            return
        self._update_stats()
        self._add_log(f"test{saved['index']}  \u2022  {res.verdict}  \u2022  {res.summary()}",
                      NG if res.verdict == "NG" else OK)
        self._last_result = ResultWindow(self, frame, res, saved, model)

    def _update_stats(self):
        model = self.current_model.get().strip()
        if not model:
            self.stats.configure(text="no captures yet")
            return
        s = self.storage.stats(model)
        self.stats.configure(text=f"{model}\ntotal {s['total']}   NG {s['ng']}   OK {s['ok']}\n"
                                  f"NG rate {s['ng_rate']:.1f}%")

    def _add_log(self, text, color):
        ts = datetime.now().strftime("%H:%M:%S")
        r = ctk.CTkLabel(self.log, text=f"{text}  \u2022  {ts}", text_color=color,
                         anchor="w", font=ctk.CTkFont(size=11))
        if self.log_rows:
            r.pack(fill="x", anchor="w", before=self.log_rows[0])
        else:
            r.pack(fill="x", anchor="w")
        self.log_rows.insert(0, r)
        if len(self.log_rows) > 12:
            self.log_rows.pop().destroy()

    # ============================================================ render
    def _autoset_min_area(self, h, w):
        if self._res_autoset_done:
            return
        self._res_autoset_done = True
        if self.current_model.get().strip():
            return
        scale = max(h, w) / 1280.0
        if scale <= 1.2:
            return
        self.min_area.set(int(round(DEFAULT_PARAMS["min_area"] * scale * scale)))
        for lbl, fmt, var in self._param_widgets.values():
            lbl.configure(text=fmt(var.get()))
        self.save_note.configure(text=f"\u2699 {max(h, w)}px camera - min size set to "
                                      f"{int(self.min_area.get())}px")
        self.after(5000, lambda: self.save_note.configure(text=""))

    def _tick(self):
        if self._pending_cams is not None:
            cams, self._pending_cams = self._pending_cams, None
            self._populate_cameras(cams)

        frame = self.camera.get_frame()
        if frame is not None:
            h, w = frame.shape[:2]
            self._frame_hw = (h, w)
            self._autoset_min_area(h, w)

            # downscale for display (the key fix for the 20MP feed hang)
            scale = min(1.0, MAX_PREVIEW / max(h, w))
            self._disp_scale = scale
            if scale < 1.0:
                disp = cv2.resize(frame, (max(int(w * scale), 1), max(int(h * scale), 1)),
                                  interpolation=cv2.INTER_LINEAR)
            else:
                disp = frame.copy()

            for i, r in enumerate(self.rois):
                color = ROI_SEL if i == self.selected_idx else ROI_COLOR
                c = (int(r["cx"] * scale), int(r["cy"] * scale))
                cv2.circle(disp, c, max(int(r["r"] * scale), 1), color, 2)
                cv2.drawMarker(disp, c, color, cv2.MARKER_CROSS, 10, 1)
                cv2.putText(disp, str(i + 1), (c[0] - 6, c[1] - max(int(r["r"] * scale), 1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            self.canvas.set_image(disp)
            self.status.configure(text=self.camera.status_text())

        self.after(40, self._tick)      # ~25 fps preview

    def _on_close(self):
        self.camera.stop()
        self.destroy()


class ResultWindow(ctk.CTkToplevel):
    def __init__(self, app, original, res, saved, model):
        super().__init__(app)
        self.app = app
        self.saved = saved
        self.model = model
        self.title(f"Result - {model} test{saved['index']}")
        self.geometry("1060x650")
        self.configure(fg_color=BG)

        col = NG if res.verdict == "NG" else OK
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=14, pady=12)
        ctk.CTkLabel(head, text=res.verdict, font=ctk.CTkFont(size=30, weight="bold"),
                     text_color=col).pack(side="left")
        ctk.CTkLabel(head, text=f"   {res.summary()}  \u2022  test{saved['index']}  \u2022  {model}",
                     text_color=MUTED, font=ctk.CTkFont(size=13)).pack(side="left", padx=10)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(body, text="Original", text_color=MUTED, font=ctk.CTkFont(size=12)
                     ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(body, text="Inspected  (dust = red, string = amber)", text_color=MUTED,
                     font=ctk.CTkFont(size=12)).grid(row=0, column=1, sticky="w")
        shared = ViewState()
        # downscale the review images too so huge frames render fast
        left = ImageCanvas(body, view_state=shared, editable=False, bg="#050608")
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        right = ImageCanvas(body, view_state=shared, editable=False, bg="#050608")
        right.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        left.set_image(self._fit(original))
        right.set_image(self._fit(res.annotated), fit_if_first=False)

        ctk.CTkLabel(self, text=f"saved: {saved['after_path']}", text_color=MUTED,
                     font=ctk.CTkFont(size=11)).pack(pady=(0, 6))
        other = "OK" if saved["verdict"] == "NG" else "NG"
        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(pady=(0, 12))
        ctk.CTkButton(btns, text="\u2713  Sahi hai - Close", width=170, fg_color=PANEL,
                      hover_color="#1b1f27", command=self.destroy).pack(side="left", padx=6)
        ctk.CTkButton(btns, text=f"\u2717  Galat - flip to {other}", width=200,
                      fg_color="#4a3a16", hover_color="#6a5320", command=self._flip
                      ).pack(side="left", padx=6)

    @staticmethod
    def _fit(img, longest=1500):
        h, w = img.shape[:2]
        s = min(1.0, longest / max(h, w))
        if s < 1.0:
            return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return img

    def _flip(self):
        old = self.saved["verdict"]
        new = "OK" if old == "NG" else "NG"
        self.app.storage.relabel(self.model, self.saved["index"], old, new)
        self.app._update_stats()
        self.app._add_log(f"test{self.saved['index']}  \u2022  overridden \u2192 {new}", ACCENT)
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
