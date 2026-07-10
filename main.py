"""
main.py
--------
Camera-module dust & string inspector.

Run:
    python main.py

Pick a camera -> click the live view to mark each lens centre -> tune
radius -> calibrate (Settings) -> Capture & Inspect. Particles are
measured in millimetres and anything under the size threshold (0.1 mm
end-to-end by default) is ignored.

UI notes:
  * The live view is DOWNSCALED for display (a 20 MP frame pushed through
    the UI 25x/sec is what used to hang it). ROIs are stored in full-res
    coordinates and inspection always runs on the full-res frame.
  * Capture shows the result INLINE - original and inspected side by side
    in the same window, pan/zoom synced. The files are still saved to
    disk exactly as before.
  * All detection settings live behind the gear button, not the sidebar.
"""

import threading
from datetime import datetime

import cv2
import customtkinter as ctk
from tkinter import messagebox, filedialog

from camera import CameraManager, enumerate_cameras
from detector import inspect, DEFAULT_PARAMS, mm_per_px_from_roi, px_for_length, white_mask_preview
from storage import Storage
from canvas_widget import ViewState, ImageCanvas

ctk.set_appearance_mode("dark")

# ---- stealth palette: black + green
BG = "#050806"
PANEL = "#0a0f0c"
PANEL2 = "#0e1512"
LINE = "#1a231d"
ACCENT = "#3ddc84"
ACCENT_DIM = "#1d6b43"
NG = "#ff5566"
OK = "#3ddc84"
MUTED = "#5c6b62"
TEXT = "#d8e6dd"

ROI_COLOR = (140, 150, 140)     # BGR unselected
ROI_SEL = (110, 220, 60)        # BGR selected (green)

MAX_PREVIEW = 1500              # longest side of the on-screen preview


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Dust Inspector")
        self.geometry("1380x880")
        self.minsize(1120, 720)
        self.configure(fg_color=BG)

        self.storage = Storage()
        self.camera = CameraManager()
        self._cam_labels = {}
        self._pending_cams = None

        self.rois = []                  # full-res coords
        self.selected_idx = None
        self.roi_rows = []
        self._disp_scale = 1.0
        self._frame_hw = (0, 0)
        self._autoset_done = False
        self._settings_win = None

        self.mode = "live"              # "live" | "review"
        self.review = None              # (original, result, saved, model)
        self.calib_mode = False         # True while picking two points for calibration
        self.calib_pts = []             # full-res points picked for the 2-point measure
        self.static_image = None        # loaded-from-disk frame; overrides live feed when set
        self.static_name = ""
        self.show_mask = False          # live mask-preview toggle

        self.current_model = ctk.StringVar(value="")
        self.sigma = ctk.DoubleVar(value=DEFAULT_PARAMS["sigma"])
        self.min_length_mm = ctk.DoubleVar(value=DEFAULT_PARAMS["min_length_mm"])
        self.mm_per_px = ctk.DoubleVar(value=DEFAULT_PARAMS["mm_per_px"])
        self.white_only = ctk.BooleanVar(value=DEFAULT_PARAMS["white_only"])
        self.white_sat = ctk.IntVar(value=DEFAULT_PARAMS["white_max_saturation"])
        self.dark_floor = ctk.IntVar(value=DEFAULT_PARAMS["dark_floor"])
        self.blur_ksize = ctk.IntVar(value=DEFAULT_PARAMS["blur_ksize"])
        self.min_pixels = ctk.IntVar(value=DEFAULT_PARAMS["min_pixels"])

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<space>", self._on_space)
        self.bind("<Escape>", lambda _e: self._back_to_live())

        threading.Thread(target=self._startup_cameras, daemon=True).start()
        self._refresh_models()
        self._tick()

    # ---------------------------------------------------------- params
    def params(self):
        return {
            "sigma": float(self.sigma.get()),
            "min_length_mm": float(self.min_length_mm.get()),
            "mm_per_px": float(self.mm_per_px.get()),
            "white_only": bool(self.white_only.get()),
            "white_max_saturation": int(self.white_sat.get()),
            "dark_floor": int(self.dark_floor.get()),
            "blur_ksize": int(self.blur_ksize.get()),
            "min_pixels": int(self.min_pixels.get()),
        }

    def _apply_params(self, p):
        self.sigma.set(p.get("sigma", DEFAULT_PARAMS["sigma"]))
        self.min_length_mm.set(p.get("min_length_mm", DEFAULT_PARAMS["min_length_mm"]))
        self.mm_per_px.set(p.get("mm_per_px", DEFAULT_PARAMS["mm_per_px"]))
        self.white_only.set(bool(p.get("white_only", DEFAULT_PARAMS["white_only"])))
        self.white_sat.set(int(p.get("white_max_saturation", DEFAULT_PARAMS["white_max_saturation"])))
        self.dark_floor.set(int(p.get("dark_floor", DEFAULT_PARAMS["dark_floor"])))
        self.blur_ksize.set(int(p.get("blur_ksize", DEFAULT_PARAMS["blur_ksize"])))
        self.min_pixels.set(int(p.get("min_pixels", DEFAULT_PARAMS["min_pixels"])))
        self._update_calib_chip()

    # ============================================================== UI
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # ---------- top bar
        top = ctk.CTkFrame(self, height=52, corner_radius=0, fg_color=PANEL)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        top.grid_propagate(False)
        ctk.CTkLabel(top, text="  \u25C9  DUST INSPECTOR", text_color=ACCENT,
                     font=ctk.CTkFont(size=15, weight="bold")).pack(side="left", padx=12, pady=10)
        self.status = ctk.CTkLabel(top, text="", text_color=MUTED, font=ctk.CTkFont(size=11))
        self.status.pack(side="left", padx=6)

        right = ctk.CTkFrame(top, fg_color="transparent")
        right.pack(side="right", padx=10)
        ctk.CTkButton(right, text="\u2699", width=40, height=30, fg_color=PANEL2,
                      hover_color=LINE, text_color=ACCENT,
                      font=ctk.CTkFont(size=16), command=self._open_settings).pack(side="right", padx=(6, 0))
        ctk.CTkButton(right, text="\u21BB", width=34, height=30, fg_color=PANEL2,
                      hover_color=LINE, text_color=TEXT,
                      command=self._rescan_cameras).pack(side="right", padx=4)
        self.cam_combo = ctk.CTkComboBox(right, width=260, height=30, values=["Scanning..."],
                                         command=self._on_camera_pick, fg_color=PANEL2,
                                         button_color=ACCENT_DIM, border_color=LINE,
                                         text_color=TEXT, dropdown_fg_color=PANEL)
        self.cam_combo.pack(side="right", padx=4)

        # source: live camera vs an opened image
        src = ctk.CTkFrame(top, fg_color="transparent")
        src.pack(side="right", padx=(0, 6))
        ctk.CTkButton(src, text="\U0001F4C1 Open Image", width=118, height=30, fg_color=PANEL2,
                      hover_color=LINE, text_color=TEXT,
                      command=self._open_image_file).pack(side="left", padx=3)
        self.live_btn = ctk.CTkButton(src, text="\u25CB Live", width=70, height=30, fg_color=PANEL2,
                                      hover_color=LINE, text_color=MUTED,
                                      command=self._use_live_feed)
        self.live_btn.pack(side="left", padx=3)

        # ---------- centre: swaps between live and review
        self.center = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.center.grid(row=1, column=0, sticky="nsew", padx=(10, 6), pady=(8, 8))
        self.center.grid_rowconfigure(1, weight=1)
        self.center.grid_columnconfigure(0, weight=1)

        # verdict banner (only in review)
        self.banner = ctk.CTkFrame(self.center, fg_color=PANEL, corner_radius=8, height=48)
        self.verdict_lbl = ctk.CTkLabel(self.banner, text="", font=ctk.CTkFont(size=22, weight="bold"))
        self.verdict_lbl.pack(side="left", padx=14, pady=8)
        self.verdict_sub = ctk.CTkLabel(self.banner, text="", text_color=MUTED,
                                        font=ctk.CTkFont(size=12))
        self.verdict_sub.pack(side="left", padx=4)
        bbtn = ctk.CTkFrame(self.banner, fg_color="transparent")
        bbtn.pack(side="right", padx=8)
        self.flip_btn = ctk.CTkButton(bbtn, text="Flip verdict", width=110, height=28,
                                      fg_color="#3a2f10", hover_color="#584918",
                                      text_color=TEXT, command=self._flip_verdict)
        self.flip_btn.pack(side="right", padx=4)
        ctk.CTkButton(bbtn, text="\u2190 Back to live", width=120, height=28, fg_color=PANEL2,
                      hover_color=LINE, text_color=TEXT,
                      command=self._back_to_live).pack(side="right", padx=4)

        # live canvas
        self.live_wrap = ctk.CTkFrame(self.center, fg_color=BG, corner_radius=0)
        self.live_wrap.grid_rowconfigure(0, weight=1)
        self.live_wrap.grid_columnconfigure(0, weight=1)
        self.view = ViewState()
        self.canvas = ImageCanvas(self.live_wrap, view_state=self.view, editable=True, bg="#02040300"[:7],
                                  on_click=self._on_click, on_radius_change=self._nudge_radius,
                                  on_delete=self._delete_selected)
        self.canvas.configure(bg="#020403")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # review canvases (side by side, synced)
        self.review_wrap = ctk.CTkFrame(self.center, fg_color=BG, corner_radius=0)
        self.review_wrap.grid_rowconfigure(1, weight=1)
        self.review_wrap.grid_columnconfigure(0, weight=1)
        self.review_wrap.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.review_wrap, text="ORIGINAL", text_color=MUTED,
                     font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=0, sticky="w", pady=(0, 3))
        ctk.CTkLabel(self.review_wrap, text="INSPECTED   \u2022  defects circled in red",
                     text_color=MUTED, font=ctk.CTkFont(size=10, weight="bold")
                     ).grid(row=0, column=1, sticky="w", pady=(0, 3))
        self.rev_view = ViewState()
        self.rev_left = ImageCanvas(self.review_wrap, view_state=self.rev_view, editable=False)
        self.rev_left.configure(bg="#020403")
        self.rev_left.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.rev_right = ImageCanvas(self.review_wrap, view_state=self.rev_view, editable=False)
        self.rev_right.configure(bg="#020403")
        self.rev_right.grid(row=1, column=1, sticky="nsew", padx=(4, 0))

        self.live_wrap.grid(row=1, column=0, sticky="nsew")

        # zoom bar
        bar = ctk.CTkFrame(self.center, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for txt, fn, w in (("\u2212", lambda: self._zoom(1 / 1.2), 32),
                           ("\u2922", self._fit, 32),
                           ("+", lambda: self._zoom(1.2), 32)):
            ctk.CTkButton(bar, text=txt, width=w, height=26, fg_color=PANEL,
                          hover_color=LINE, text_color=TEXT, command=fn).pack(side="left", padx=3)
        self.mask_btn = ctk.CTkButton(bar, text="\u25D1 Mask view", width=104, height=26,
                                      fg_color=PANEL, hover_color=LINE, text_color=MUTED,
                                      command=self._toggle_mask)
        self.mask_btn.pack(side="left", padx=(10, 3))
        self.hint = ctk.CTkLabel(bar, text="click = ROI   drag = pan   scroll = zoom   "
                                           "shift+scroll = radius   del = remove   space = inspect",
                                 text_color=MUTED, font=ctk.CTkFont(size=10))
        self.hint.pack(side="left", padx=12)

        # calibration strip (hidden unless picking two points)
        self.calib_bar = ctk.CTkFrame(self.center, fg_color="#10251a", corner_radius=6)
        self.calib_status = ctk.CTkLabel(self.calib_bar, text="", text_color=ACCENT,
                                         font=ctk.CTkFont(size=11, weight="bold"))
        self.calib_status.pack(side="left", padx=(12, 8), pady=5)
        ctk.CTkButton(self.calib_bar, text="\u21B6 Undo point", width=100, height=26,
                      fg_color=PANEL2, hover_color=LINE, text_color=TEXT,
                      command=self._undo_two_point).pack(side="left", padx=3, pady=5)
        ctk.CTkButton(self.calib_bar, text="\u2715 Cancel", width=84, height=26,
                      fg_color="#3a1418", hover_color="#5c2027", text_color=TEXT,
                      command=self._cancel_two_point).pack(side="left", padx=3, pady=5)

        # ---------- sidebar
        side = ctk.CTkFrame(self, width=316, fg_color=PANEL, corner_radius=0)
        side.grid(row=1, column=1, sticky="ns", pady=(8, 8), padx=(0, 10))
        side.grid_propagate(False)
        self._build_sidebar(side)

    def _cap(self, parent, text):
        ctk.CTkLabel(parent, text=text.upper(), text_color=ACCENT_DIM,
                     font=ctk.CTkFont(size=10, weight="bold")).pack(anchor="w", padx=10, pady=(14, 4))

    def _build_sidebar(self, p):
        # model
        self._cap(p, "model")
        self.model_combo = ctk.CTkComboBox(p, values=[], variable=self.current_model, height=30,
                                           command=lambda _c: self._load_model(), fg_color=PANEL2,
                                           button_color=ACCENT_DIM, border_color=LINE,
                                           text_color=TEXT, dropdown_fg_color=PANEL)
        self.model_combo.pack(fill="x", padx=10)
        row = ctk.CTkFrame(p, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(row, text="Load", height=28, fg_color=PANEL2, hover_color=LINE,
                      text_color=TEXT, command=self._load_model).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ctk.CTkButton(row, text="Save", height=28, fg_color=ACCENT, hover_color=TEXT,
                      text_color="#04160c", command=self._save_model).pack(side="left", expand=True, fill="x", padx=3)
        ctk.CTkButton(row, text="\u2715", width=34, height=28, fg_color="#3a1418",
                      hover_color="#5c2027", text_color=TEXT,
                      command=self._delete_model).pack(side="left", padx=(3, 0))
        self.save_note = ctk.CTkLabel(p, text="", text_color=OK, font=ctk.CTkFont(size=10))
        self.save_note.pack(anchor="w", padx=10)

        # calibration chip
        self.calib_chip = ctk.CTkLabel(p, text="", text_color=MUTED, font=ctk.CTkFont(size=10),
                                       anchor="w", justify="left")
        self.calib_chip.pack(fill="x", padx=10, pady=(6, 0))

        # ROIs
        self._cap(p, "lens ROIs")
        self.roi_frame = ctk.CTkFrame(p, fg_color="transparent")
        self.roi_frame.pack(fill="x", padx=10)
        self.radius = ctk.CTkSlider(p, from_=10, to=1600, progress_color=ACCENT, button_color=ACCENT,
                                    button_hover_color=TEXT, height=16, command=self._radius_slider)
        self.radius.set(120)
        self.radius.pack(fill="x", padx=10, pady=(8, 0))
        self.radius_note = ctk.CTkLabel(p, text="select an ROI to edit radius",
                                        text_color=MUTED, font=ctk.CTkFont(size=10))
        self.radius_note.pack(anchor="w", padx=10)
        self._rebuild_roi_list()

        # capture
        ctk.CTkButton(p, text="CAPTURE & INSPECT", height=44, fg_color=ACCENT, hover_color=TEXT,
                      text_color="#04160c", font=ctk.CTkFont(size=13, weight="bold"),
                      command=self._capture).pack(fill="x", padx=10, pady=(16, 4))
        self.reopen_btn = ctk.CTkButton(p, text="Open last result side-by-side", height=26,
                                        fg_color=PANEL2, hover_color=LINE, text_color=MUTED,
                                        font=ctk.CTkFont(size=11), state="disabled",
                                        command=self._reopen_review)
        self.reopen_btn.pack(fill="x", padx=10)

        # detected particles
        self._cap(p, "detected particles")
        self.defects_box = ctk.CTkScrollableFrame(p, height=136, fg_color=PANEL2,
                                                  scrollbar_button_color=LINE)
        self.defects_box.pack(fill="x", padx=10)
        self._set_defect_list(None)

        # stats
        self._cap(p, "stats")
        self.stats = ctk.CTkLabel(p, text="no captures yet", justify="left", anchor="w",
                                  text_color=TEXT, font=ctk.CTkFont(size=11))
        self.stats.pack(fill="x", padx=10)

        # log
        self._cap(p, "log")
        self.logbox = ctk.CTkTextbox(p, height=132, fg_color=PANEL2, text_color=MUTED,
                                     font=ctk.CTkFont(family="Courier", size=10),
                                     border_width=0, activate_scrollbars=True)
        self.logbox.pack(fill="x", padx=10, pady=(0, 12))
        self.logbox.configure(state="disabled")
        self._log("ready")

    # ============================================================ logging
    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.logbox.configure(state="normal")
        self.logbox.insert("1.0", f"{ts}  {msg}\n")
        self.logbox.configure(state="disabled")

    # ========================================================= calibration
    def _update_calib_chip(self):
        mmpp = float(self.mm_per_px.get())
        if mmpp > 0:
            px = px_for_length(float(self.min_length_mm.get()), mmpp)
            self.calib_chip.configure(
                text=f"\u25CF calibrated  {mmpp * 1000:.2f} \u00B5m/px   \u2022   "
                     f"{self.min_length_mm.get():.2f} mm = {px:.1f} px",
                text_color=OK)
        else:
            self.calib_chip.configure(
                text="\u25CB not calibrated - sizes in px. Set it in \u2699 Settings.",
                text_color=MUTED)

    # ============================================================ cameras
    def _startup_cameras(self):
        self._pending_cams = self.camera.connect_best_available()

    def _rescan_cameras(self):
        self.cam_combo.configure(values=["Scanning..."])
        self.cam_combo.set("Scanning...")
        threading.Thread(target=lambda: setattr(self, "_pending_cams", enumerate_cameras()),
                         daemon=True).start()

    def _populate_cameras(self, cams):
        self._cam_labels = {c["name"]: c for c in cams}
        self.cam_combo.configure(values=[c["name"] for c in cams] + ["Synthetic feed (no camera)"])
        self.cam_combo.set(self.camera.active["name"] if self.camera.active
                           else "Synthetic feed (no camera)")
        self._log(f"cameras found: {len(cams)}")

    def _on_camera_pick(self, label):
        desc = self._cam_labels.get(label)
        self.cam_combo.set(label)
        self._log(f"connecting: {label}")
        threading.Thread(target=lambda: self.camera.connect(desc), daemon=True).start()

    # ---------------------------------------------------------- image source
    def frame_source(self):
        """The frame the whole app works on: a loaded image if one is open,
        otherwise the live camera. ROIs, calibration, capture all use this."""
        if self.static_image is not None:
            return self.static_image
        return self.camera.get_frame()

    def _open_image_file(self):
        path = filedialog.askopenfilename(
            title="Open an already-captured image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Load failed", f"Ye file image ki tarah nahi khuli:\n{path}")
            return
        import os as _os
        self.static_image = img
        self.static_name = _os.path.basename(path)
        self._autoset_done = False          # re-evaluate noise floor for this image's size
        self.calib_pts = []                 # any half-done calibration is now stale
        self.calib_mode = False
        self.live_btn.configure(text_color=TEXT)
        self._back_to_live()
        self.canvas.source_image = None     # force a fresh fit to the new image
        self._log(f"opened image: {self.static_name}  ({img.shape[1]}x{img.shape[0]})")

    def _use_live_feed(self):
        if self.static_image is None:
            return
        self.static_image = None
        self.static_name = ""
        self.calib_pts = []
        self.calib_mode = False
        self.live_btn.configure(text_color=MUTED)
        self.canvas.source_image = None
        self._back_to_live()
        self._log("switched back to live feed")

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
        self._log(f"loaded model '{model}' ({len(self.rois)} ROI)")

    def _save_model(self):
        model = self.current_model.get().strip()
        if not model:
            messagebox.showwarning("Model name needed", "Pehle model ka naam likho ya select karo.")
            return
        self.storage.save_config(model, self.rois, self.params())
        self._refresh_models()
        self.current_model.set(model)
        self._update_stats()
        self.save_note.configure(text=f"\u2713 saved '{model}'")
        self.after(2000, lambda: self.save_note.configure(text=""))
        self._log(f"saved model '{model}'")

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
        self._log(f"deleted config '{model}'")

    # ============================================================ ROI edit
    def _on_click(self, nx, ny, cx, cy):
        if self.mode != "live":
            return
        fx, fy = nx / self._disp_scale, ny / self._disp_scale

        if self.calib_mode:
            self.calib_pts.append((fx, fy))
            if len(self.calib_pts) >= 2:
                self.calib_mode = False
                (x1, y1), (x2, y2) = self.calib_pts[0], self.calib_pts[1]
                dist_px = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                self._log(f"2-point measure: {dist_px:.1f} px")
                if self._settings_win is not None and self._settings_win.winfo_exists():
                    self._settings_win.two_points_done(dist_px)
            self._update_calib_bar()
            return

        hit = self._hit(cx, cy)
        if hit is not None:
            self.selected_idx = hit
        else:
            base = min(self._frame_hw) if min(self._frame_hw) > 0 else 960
            self.rois.append({"cx": fx, "cy": fy, "r": max(20, int(base * 0.04))})
            self.selected_idx = len(self.rois) - 1
        self._rebuild_roi_list()

    def _show_calib_bar(self):
        self.calib_bar.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        self._update_calib_bar()

    def _hide_calib_bar(self):
        self.calib_bar.grid_forget()

    def _update_calib_bar(self):
        n = len(self.calib_pts)
        if n == 0:
            self.calib_status.configure(text="CALIBRATION  \u2022  click point 1 of 2")
        elif n == 1:
            self.calib_status.configure(text="CALIBRATION  \u2022  click point 2 of 2")
        else:
            (x1, y1), (x2, y2) = self.calib_pts[0], self.calib_pts[1]
            d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            self.calib_status.configure(text=f"CALIBRATION  \u2022  {d:.1f} px measured  "
                                             f"\u2192 enter mm in Settings")

    def start_two_point(self):
        self.calib_mode = True
        self.calib_pts = []
        self._back_to_live()
        self._show_calib_bar()
        self.hint.configure(text="CALIBRATION: click the two ends of a known length   "
                                 "\u2022   use the buttons above to undo / cancel")
        self._log("2-point measure: click two points (Undo / Cancel available)")

    def _cancel_two_point(self):
        self.calib_mode = False
        self.calib_pts = []
        self._hide_calib_bar()
        self.hint.configure(text="click = ROI   drag = pan   scroll = zoom   "
                                 "shift+scroll = radius   del = remove   space = inspect")
        self._log("calibration cancelled")
        if self._settings_win is not None and self._settings_win.winfo_exists():
            self._settings_win.two_pt_lbl.configure(text="cancelled", text_color=MUTED)
            self._settings_win.lift()

    def _undo_two_point(self):
        if self.calib_pts:
            self.calib_pts.pop()
            self._log(f"undo point ({len(self.calib_pts)} left)")
        if not self.calib_mode:                 # re-enter picking if a full pair was undone
            self.calib_mode = True
            self._show_calib_bar()
        self._update_calib_bar()

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
        self.rois[self.selected_idx]["r"] = max(4, self.rois[self.selected_idx]["r"] + delta / self._disp_scale)
        self._rebuild_roi_list()

    def _radius_slider(self, value):
        if self.selected_idx is None or self.selected_idx >= len(self.rois):
            return
        self.rois[self.selected_idx]["r"] = float(value)
        self.radius_note.configure(text=f"ROI #{self.selected_idx + 1}   r = {int(float(value))} px")
        if self.selected_idx < len(self.roi_rows):
            self.roi_rows[self.selected_idx].configure(text=f"#{self.selected_idx + 1}   r = {int(float(value))}px")

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
                         text_color=MUTED, font=ctk.CTkFont(size=10)).pack(anchor="w", pady=2)
        for i, r in enumerate(self.rois):
            sel = i == self.selected_idx
            rowf = ctk.CTkFrame(self.roi_frame, fg_color=("#10251a" if sel else "transparent"),
                                corner_radius=5)
            rowf.pack(fill="x", pady=1)
            btn = ctk.CTkButton(rowf, text=f"#{i + 1}   r = {int(r['r'])}px", anchor="w", height=26,
                                fg_color="transparent", text_color=(ACCENT if sel else TEXT),
                                hover_color=LINE, font=ctk.CTkFont(size=11),
                                command=lambda idx=i: self._select(idx))
            btn.pack(side="left", expand=True, fill="x")
            self.roi_rows.append(btn)
            ctk.CTkButton(rowf, text="\u2715", width=24, height=26, fg_color="transparent",
                          hover_color="#5c2027", text_color=MUTED,
                          command=lambda idx=i: self._delete_at(idx)).pack(side="right")
        if self.selected_idx is not None and self.selected_idx < len(self.rois):
            self.radius.set(min(self.rois[self.selected_idx]["r"], 1600))
            self.radius_note.configure(
                text=f"ROI #{self.selected_idx + 1}   r = {int(self.rois[self.selected_idx]['r'])} px")
        else:
            self.radius_note.configure(text="select an ROI to edit radius")

    # =============================================================== view
    def _active_view(self):
        return self.view if self.mode == "live" else self.rev_view

    def _active_canvas(self):
        return self.canvas if self.mode == "live" else self.rev_left

    def _zoom(self, f):
        c = self._active_canvas()
        self._active_view().zoom_at(c.winfo_width() / 2, c.winfo_height() / 2, f)

    def _toggle_mask(self):
        self.show_mask = not self.show_mask
        self.mask_btn.configure(text_color=(ACCENT if self.show_mask else MUTED))
        self._log("mask view " + ("on - coloured/dark pixels blacked out" if self.show_mask else "off"))

    def _fit(self):
        self._active_canvas().fit_to_window()

    # ============================================================ capture
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
        frame = self.frame_source()          # FULL resolution
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
        self._log(f"test{saved['index']}  {res.verdict}  {res.summary()}")
        self.review = (frame, res, saved, model)
        self.reopen_btn.configure(state="normal", text_color=TEXT)
        self._show_review()

    def _show_review(self):
        if self.review is None:
            return
        frame, res, saved, model = self.review
        self.mode = "review"
        self.live_wrap.grid_forget()
        self.banner.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.review_wrap.grid(row=1, column=0, sticky="nsew")

        col = NG if res.verdict == "NG" else OK
        self.verdict_lbl.configure(text=res.verdict, text_color=col)
        self.verdict_sub.configure(text=f"{res.summary()}   \u2022   test{saved['index']}   \u2022   {model}")
        other = "OK" if saved["verdict"] == "NG" else "NG"
        self.flip_btn.configure(text=f"Flip \u2192 {other}")

        self.rev_view.zoom = 1.0
        self.rev_view.pan_x = self.rev_view.pan_y = 0.0
        self.rev_left.source_image = None
        self.rev_right.source_image = None
        self.rev_left.set_image(self._fit_img(frame))
        self.rev_right.set_image(self._fit_img(res.annotated), fit_if_first=False)

        self._set_defect_list(res)
        self.hint.configure(text="side-by-side \u2022 pan/zoom synced \u2022 Esc = back to live")

    def _reopen_review(self):
        if self.review:
            self._show_review()

    def _back_to_live(self):
        if self.mode == "live":
            return
        self.mode = "live"
        self.banner.grid_forget()
        self.review_wrap.grid_forget()
        self.live_wrap.grid(row=1, column=0, sticky="nsew")
        self.hint.configure(text="click = ROI   drag = pan   scroll = zoom   "
                                 "shift+scroll = radius   del = remove   space = inspect")

    def _flip_verdict(self):
        if self.review is None:
            return
        _f, res, saved, model = self.review
        old = saved["verdict"]
        new = "OK" if old == "NG" else "NG"
        self.storage.relabel(model, saved["index"], old, new)
        saved["verdict"] = new
        self._update_stats()
        self._log(f"test{saved['index']}  overridden -> {new}")
        col = NG if new == "NG" else OK
        self.verdict_lbl.configure(text=new, text_color=col)
        self.flip_btn.configure(text=f"Flip \u2192 {'OK' if new == 'NG' else 'NG'}")

    @staticmethod
    def _fit_img(img, longest=1500):
        h, w = img.shape[:2]
        s = min(1.0, longest / max(h, w))
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else img

    def _set_defect_list(self, res):
        for w in self.defects_box.winfo_children():
            w.destroy()
        if res is None or not res.defects:
            ctk.CTkLabel(self.defects_box, text="none" if res else "-", text_color=MUTED,
                         font=ctk.CTkFont(size=10)).pack(anchor="w")
            return
        order = sorted(res.defects, key=lambda b: b["feret_px"], reverse=True)
        for i, b in enumerate(order[:14]):
            dot = "\u25CF"
            col = "#ff5566"
            if res.calibrated:
                txt = f"{dot} {b['feret_mm']:.3f} mm   {b['area_mm2']:.4f} mm\u00B2"
            else:
                txt = f"{dot} {b['feret_px']:.0f} px   {b['area_px']:.0f} px\u00B2"
            ctk.CTkLabel(self.defects_box, text=txt, text_color=col, anchor="w",
                         font=ctk.CTkFont(family="Courier", size=10)).pack(anchor="w")
        if len(order) > 14:
            ctk.CTkLabel(self.defects_box, text=f"  +{len(order) - 14} more", text_color=MUTED,
                         font=ctk.CTkFont(size=10)).pack(anchor="w")

    def _update_stats(self):
        model = self.current_model.get().strip()
        if not model:
            self.stats.configure(text="no captures yet")
            return
        s = self.storage.stats(model)
        self.stats.configure(text=f"{model}\ntotal {s['total']}    NG {s['ng']}    OK {s['ok']}\n"
                                  f"NG rate {s['ng_rate']:.1f}%")

    # =========================================================== settings
    def _open_settings(self):
        if self._settings_win is not None and self._settings_win.winfo_exists():
            self._settings_win.lift()
            return
        self._settings_win = SettingsWindow(self)

    # ============================================================ render
    def _autoset(self, h, w):
        """Only used when UNCALIBRATED: a 20MP frame's coating grain forms
        clusters of tens of pixels, so a 4px noise floor floods the result.
        Once a mm calibration exists the physical size rule takes over and
        this is irrelevant."""
        if self._autoset_done:
            return
        self._autoset_done = True
        if self.current_model.get().strip() or float(self.mm_per_px.get()) > 0:
            return
        scale = max(h, w) / 1280.0
        if scale <= 1.2:
            return
        self.min_pixels.set(int(round(DEFAULT_PARAMS["min_pixels"] * scale * scale)))
        self._log(f"{max(h, w)}px camera - noise floor set to {self.min_pixels.get()}px "
                  f"(calibrate to use mm instead)")

    def _tick(self):
        if self._pending_cams is not None:
            cams, self._pending_cams = self._pending_cams, None
            self._populate_cameras(cams)

        frame = self.frame_source()
        if frame is not None:
            h, w = frame.shape[:2]
            self._frame_hw = (h, w)
            self._autoset(h, w)
            if self.static_image is not None:
                self.status.configure(text=f"\U0001F5BC {self.static_name}   {w}x{h}")
            else:
                self.status.configure(text=self.camera.status_text())

            if self.mode == "live":
                scale = min(1.0, MAX_PREVIEW / max(h, w))
                self._disp_scale = scale
                src = frame
                if self.show_mask and self.rois:
                    try:
                        src = white_mask_preview(frame, self.rois, self.params())
                    except Exception:
                        src = frame
                disp = (cv2.resize(src, (max(int(w * scale), 1), max(int(h * scale), 1)),
                                   interpolation=cv2.INTER_LINEAR) if scale < 1.0 else src.copy())
                for i, r in enumerate(self.rois):
                    color = ROI_SEL if i == self.selected_idx else ROI_COLOR
                    c = (int(r["cx"] * scale), int(r["cy"] * scale))
                    rr = max(int(r["r"] * scale), 1)
                    cv2.circle(disp, c, rr, color, 2)
                    cv2.drawMarker(disp, c, color, cv2.MARKER_CROSS, 9, 1)
                    cv2.putText(disp, str(i + 1), (c[0] - 5, c[1] - rr - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                # calibration measuring line
                if self.calib_pts:
                    pts = [(int(x * scale), int(y * scale)) for x, y in self.calib_pts]
                    for pt in pts:
                        cv2.drawMarker(disp, pt, (60, 230, 120), cv2.MARKER_TILTED_CROSS, 14, 2)
                    if len(pts) == 2:
                        cv2.line(disp, pts[0], pts[1], (60, 230, 120), 2)

                self.canvas.set_image(disp)

        self.after(40, self._tick)

    def _on_close(self):
        self.camera.stop()
        self.destroy()


# ======================================================================
class SettingsWindow(ctk.CTkToplevel):
    """Everything tunable lives here, out of the main window."""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Settings")
        self.geometry("500x780")
        self.configure(fg_color=BG)
        self.transient(app)

        wrap = ctk.CTkScrollableFrame(self, fg_color=BG, scrollbar_button_color=LINE)
        wrap.pack(fill="both", expand=True, padx=10, pady=10)

        # ---------------- calibration
        self._cap(wrap, "calibration  (pixels \u2192 millimetres)")
        ctk.CTkLabel(wrap, text="An image can't reveal its own scale: 20 MP says nothing\n"
                               "about how wide a scene it covers. Give it ONE real-world\n"
                               "reference below - any single method is enough.\n\n"
                               "It is ONE scale for the whole image, not per-ROI. Every\n"
                               "lens you circle inherits it. Re-do it only if the camera\n"
                               "height or lens changes.",
                     text_color=MUTED, justify="left",
                     font=ctk.CTkFont(size=10)).pack(anchor="w", padx=10, pady=(0, 6))

        # --- A: two-point measure (most reliable)
        rowa = ctk.CTkFrame(wrap, fg_color=PANEL2, corner_radius=8)
        rowa.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(rowa, text="A.  Measure two points   (recommended)", text_color=TEXT,
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=10, pady=(8, 0))
        ctk.CTkLabel(rowa, text="Put a ruler / caliper / any part of known length in view,\n"
                               "click its two ends, then type the true distance.",
                     text_color=MUTED, justify="left",
                     font=ctk.CTkFont(size=9)).pack(anchor="w", padx=10, pady=(0, 4))
        ra = ctk.CTkFrame(rowa, fg_color="transparent")
        ra.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkButton(ra, text="Pick 2 points", width=104, height=28, fg_color=ACCENT_DIM,
                      hover_color=ACCENT, text_color=TEXT,
                      command=self._start_two_point).pack(side="left")
        self.two_pt_lbl = ctk.CTkLabel(ra, text="\u2014", text_color=MUTED,
                                       font=ctk.CTkFont(family="Courier", size=10))
        self.two_pt_lbl.pack(side="left", padx=8)
        self.two_pt_mm = ctk.CTkEntry(ra, width=62, height=28, fg_color=BG, border_color=LINE,
                                      text_color=TEXT, placeholder_text="mm")
        self.two_pt_mm.pack(side="left", padx=(0, 4))
        ctk.CTkButton(ra, text="Apply", width=58, height=28, fg_color=ACCENT, hover_color=TEXT,
                      text_color="#04160c", command=self._calib_two_point).pack(side="right")
        self._two_pt_px = 0.0

        # --- B: optics calculator
        rowb = ctk.CTkFrame(wrap, fg_color=PANEL2, corner_radius=8)
        rowb.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(rowb, text="B.  From optics", text_color=TEXT,
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=10, pady=(8, 0))
        ctk.CTkLabel(rowb, text="Sensor pixel size, lens focal length, and the distance\n"
                               "from lens to phone.  mm/px = px_size \u00D7 (WD \u2212 f) / f",
                     text_color=MUTED, justify="left",
                     font=ctk.CTkFont(size=9)).pack(anchor="w", padx=10, pady=(0, 4))
        rb = ctk.CTkFrame(rowb, fg_color="transparent")
        rb.pack(fill="x", padx=10, pady=(0, 8))
        self.px_um = ctk.CTkEntry(rb, width=56, height=28, fg_color=BG, border_color=LINE,
                                  text_color=TEXT, placeholder_text="2.4")
        self.px_um.insert(0, "2.4")
        self.px_um.pack(side="left")
        ctk.CTkLabel(rb, text="\u00B5m", text_color=MUTED, font=ctk.CTkFont(size=10)).pack(side="left", padx=(2, 6))
        self.focal = ctk.CTkEntry(rb, width=52, height=28, fg_color=BG, border_color=LINE,
                                  text_color=TEXT, placeholder_text="50")
        self.focal.insert(0, "50")
        self.focal.pack(side="left")
        ctk.CTkLabel(rb, text="f mm", text_color=MUTED, font=ctk.CTkFont(size=10)).pack(side="left", padx=(2, 6))
        self.wd = ctk.CTkEntry(rb, width=58, height=28, fg_color=BG, border_color=LINE,
                               text_color=TEXT, placeholder_text="WD mm")
        self.wd.pack(side="left")
        ctk.CTkLabel(rb, text="WD", text_color=MUTED, font=ctk.CTkFont(size=10)).pack(side="left", padx=(2, 4))
        ctk.CTkButton(rb, text="Apply", width=58, height=28, fg_color=ACCENT, hover_color=TEXT,
                      text_color="#04160c", command=self._calib_optics).pack(side="right")

        # --- C: field of view width
        rowc = ctk.CTkFrame(wrap, fg_color=PANEL2, corner_radius=8)
        rowc.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(rowc, text="C.  From field of view", text_color=TEXT,
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=10, pady=(8, 0))
        ctk.CTkLabel(rowc, text="How many mm the FULL image width covers.",
                     text_color=MUTED, font=ctk.CTkFont(size=9)).pack(anchor="w", padx=10, pady=(0, 4))
        rc = ctk.CTkFrame(rowc, fg_color="transparent")
        rc.pack(fill="x", padx=10, pady=(0, 8))
        self.fov_mm = ctk.CTkEntry(rc, width=80, height=28, fg_color=BG, border_color=LINE,
                                   text_color=TEXT, placeholder_text="65.7")
        self.fov_mm.pack(side="left")
        ctk.CTkLabel(rc, text="mm across the image", text_color=MUTED,
                     font=ctk.CTkFont(size=10)).pack(side="left", padx=6)
        ctk.CTkButton(rc, text="Apply", width=58, height=28, fg_color=ACCENT, hover_color=TEXT,
                      text_color="#04160c", command=self._calib_fov).pack(side="right")

        # --- D: from an ROI of known real diameter
        rowd = ctk.CTkFrame(wrap, fg_color=PANEL2, corner_radius=8)
        rowd.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(rowd, text="D.  From one reference ROI", text_color=TEXT,
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=10, pady=(8, 0))
        ctk.CTkLabel(rowd, text="Pick ONE circled part whose real diameter you know\n"
                               "(e.g. the main lens ring). The other ROIs need nothing.",
                     text_color=MUTED, justify="left",
                     font=ctk.CTkFont(size=9)).pack(anchor="w", padx=10, pady=(0, 4))
        rd = ctk.CTkFrame(rowd, fg_color="transparent")
        rd.pack(fill="x", padx=10, pady=(0, 8))
        self.roi_pick = ctk.CTkComboBox(rd, width=84, height=28, values=self._roi_names(),
                                        fg_color=BG, button_color=ACCENT_DIM, border_color=LINE,
                                        text_color=TEXT, dropdown_fg_color=PANEL)
        self.roi_pick.pack(side="left")
        if self._roi_names():
            self.roi_pick.set(self._roi_names()[0])
        ctk.CTkLabel(rd, text="\u00D8", text_color=MUTED,
                     font=ctk.CTkFont(size=12)).pack(side="left", padx=(8, 4))
        self.real_dia = ctk.CTkEntry(rd, width=62, height=28, fg_color=BG, border_color=LINE,
                                     text_color=TEXT, placeholder_text="8.0")
        self.real_dia.pack(side="left")
        ctk.CTkLabel(rd, text="mm", text_color=MUTED, font=ctk.CTkFont(size=10)).pack(side="left", padx=4)
        ctk.CTkButton(rd, text="Apply", width=58, height=28, fg_color=ACCENT, hover_color=TEXT,
                      text_color="#04160c", command=self._calib_from_roi).pack(side="right")

        self.calib_out = ctk.CTkLabel(wrap, text="", text_color=OK, justify="left", anchor="w",
                                      font=ctk.CTkFont(family="Courier", size=11))
        self.calib_out.pack(fill="x", padx=10, pady=(8, 0))
        ctk.CTkButton(wrap, text="Clear calibration (back to pixel mode)", height=26,
                      fg_color=PANEL2, hover_color=LINE, text_color=MUTED,
                      font=ctk.CTkFont(size=10), command=self._clear_calib).pack(fill="x", padx=10, pady=4)

        # ---------------- size threshold
        self._cap(wrap, "size threshold")
        ctk.CTkLabel(wrap, text="A particle counts only if its longest end-to-end span\n"
                               "reaches this. For a round particle that is its diameter.",
                     text_color=MUTED, justify="left",
                     font=ctk.CTkFont(size=10)).pack(anchor="w", padx=10, pady=(0, 4))
        r3 = ctk.CTkFrame(wrap, fg_color="transparent")
        r3.pack(fill="x", padx=10)
        self.mm_entry = ctk.CTkEntry(r3, width=90, height=28, fg_color=PANEL2, border_color=LINE,
                                     text_color=TEXT)
        self.mm_entry.insert(0, f"{app.min_length_mm.get():.3f}")
        self.mm_entry.pack(side="left")
        ctk.CTkLabel(r3, text="mm  (min span)", text_color=MUTED,
                     font=ctk.CTkFont(size=11)).pack(side="left", padx=6)
        ctk.CTkButton(r3, text="Set", width=54, height=28, fg_color=ACCENT, hover_color=TEXT,
                      text_color="#04160c", command=self._set_mm).pack(side="right")

        # ---------------- detection
        self._cap(wrap, "detection")
        self._slider(wrap, "Sensitivity (Z-score)   lower = stricter",
                     app.sigma, 2.0, 6.0, lambda v: f"{v:.1f}")
        self._slider(wrap, "White strictness   lower = only pure white",
                     app.white_sat, 20, 180, lambda v: str(int(v)))
        self._slider(wrap, "Dark floor   gray below this is masked out (0 = off)",
                     app.dark_floor, 0, 150, lambda v: str(int(v)))
        self._slider(wrap, "Noise floor (px)   used when uncalibrated",
                     app.min_pixels, 1, 400, lambda v: str(int(v)))
        sw = ctk.CTkFrame(wrap, fg_color="transparent")
        sw.pack(fill="x", padx=10, pady=(8, 0))
        ctk.CTkSwitch(sw, text="White defects only (ignore colour)", variable=app.white_only,
                      progress_color=ACCENT, text_color=TEXT,
                      font=ctk.CTkFont(size=11)).pack(anchor="w")

        r4 = ctk.CTkFrame(wrap, fg_color="transparent")
        r4.pack(fill="x", padx=10, pady=(10, 0))
        ctk.CTkLabel(r4, text="Pre-blur", text_color=TEXT, font=ctk.CTkFont(size=11)).pack(side="left")
        self.blur_pick = ctk.CTkComboBox(r4, width=110, height=28, values=["off", "3 px", "5 px"],
                                         fg_color=PANEL2, button_color=ACCENT_DIM, border_color=LINE,
                                         text_color=TEXT, dropdown_fg_color=PANEL,
                                         command=self._set_blur)
        kb = int(app.blur_ksize.get())
        self.blur_pick.set("off" if kb < 3 else f"{kb} px")
        self.blur_pick.pack(side="right")
        ctk.CTkLabel(wrap, text="Blur only helps FIND faint particles; they are always\n"
                               "MEASURED on the unblurred image, so size stays exact.",
                     text_color=MUTED, justify="left",
                     font=ctk.CTkFont(size=10)).pack(anchor="w", padx=10, pady=(4, 0))

        ctk.CTkButton(wrap, text="Done", height=34, fg_color=ACCENT, hover_color=TEXT,
                      text_color="#04160c", font=ctk.CTkFont(size=12, weight="bold"),
                      command=self.destroy).pack(fill="x", padx=10, pady=(18, 6))
        ctk.CTkLabel(wrap, text="Settings apply immediately. Hit Save on the model\n"
                               "to persist them for that phone.",
                     text_color=MUTED, justify="left",
                     font=ctk.CTkFont(size=10)).pack(anchor="w", padx=10)

        self._refresh_calib_out()

    # ------------------------------------------------------------ helpers
    def _cap(self, parent, text):
        ctk.CTkLabel(parent, text=text.upper(), text_color=ACCENT_DIM,
                     font=ctk.CTkFont(size=10, weight="bold")).pack(anchor="w", padx=10, pady=(14, 4))

    def _slider(self, parent, title, var, lo, hi, fmt):
        ctk.CTkLabel(parent, text=title, text_color=TEXT,
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=10, pady=(8, 0))
        val = ctk.CTkLabel(parent, text=fmt(var.get()), text_color=MUTED, font=ctk.CTkFont(size=10))
        ctk.CTkSlider(parent, from_=lo, to=hi, variable=var, progress_color=ACCENT,
                      button_color=ACCENT, button_hover_color=TEXT, height=16,
                      command=lambda v, l=val, f=fmt: l.configure(text=f(float(v)))
                      ).pack(fill="x", padx=10)
        val.pack(anchor="e", padx=10)

    def _roi_names(self):
        return [f"ROI #{i + 1}" for i in range(len(self.app.rois))]

    def _refresh_calib_out(self):
        mmpp = float(self.app.mm_per_px.get())
        if mmpp > 0:
            px = px_for_length(float(self.app.min_length_mm.get()), mmpp)
            h, w = self.app._frame_hw
            fov = f"\nfield of view    =  {w * mmpp:.1f} x {h * mmpp:.1f} mm" if w else ""
            self.calib_out.configure(
                text=f"scale            =  {mmpp * 1000:.3f} \u00B5m/px\n"
                     f"{self.app.min_length_mm.get():.3f} mm threshold =  {px:.1f} px\n"
                     f"size resolution  =  {mmpp * 1000:.2f} \u00B5m (1 px){fov}",
                text_color=OK)
        else:
            self.calib_out.configure(text="not calibrated - sizes reported in pixels",
                                     text_color=MUTED)
        self.app._update_calib_chip()

    # ------------------------------------------------------------ actions
    def _start_two_point(self):
        self._two_pt_px = 0.0
        self.two_pt_lbl.configure(text="click 2 pts\u2026", text_color=ACCENT)
        self.app.start_two_point()

    def two_points_done(self, dist_px):
        """Called back by the app once the operator has clicked both points."""
        self._two_pt_px = float(dist_px)
        self.two_pt_lbl.configure(text=f"{dist_px:.1f} px", text_color=OK)
        self.lift()

    def _calib_two_point(self):
        if self._two_pt_px <= 0:
            messagebox.showwarning("Pick points first",
                                   "Pehle 'Pick 2 points' dabao, phir image pe do point click karo.",
                                   parent=self)
            return
        try:
            mm = float(self.two_pt_mm.get())
        except ValueError:
            messagebox.showwarning("Check input", "Un do points ke beech ki asli doori (mm) daalo.",
                                   parent=self)
            return
        if mm <= 0:
            return
        self._set_mmpp(mm / self._two_pt_px, f"2-point: {mm} mm over {self._two_pt_px:.1f} px")

    def _calib_optics(self):
        """mm/px = pixel_size x (WD - f) / f.

        Thin-lens: magnification m = f / (WD - f), and one pixel on the
        sensor maps to pixel_size / m on the object. WD is lens-to-object.
        """
        try:
            px_um = float(self.px_um.get())
            f = float(self.focal.get())
            wd = float(self.wd.get())
        except ValueError:
            messagebox.showwarning("Check input", "Pixel size (\u00B5m), focal length (mm) aur "
                                                   "working distance (mm) daalo.", parent=self)
            return
        if px_um <= 0 or f <= 0 or wd <= f:
            messagebox.showwarning("Check input",
                                   "Working distance focal length se zyada honi chahiye.", parent=self)
            return
        mag = f / (wd - f)
        mmpp = (px_um / 1000.0) / mag
        self._set_mmpp(mmpp, f"optics: {px_um}\u00B5m px, f={f}mm, WD={wd}mm (mag {mag:.3f}x)")

    def _calib_fov(self):
        w = self.app._frame_hw[1]
        if w <= 0:
            messagebox.showwarning("No frame", "Camera se frame aane do pehle.", parent=self)
            return
        try:
            fov = float(self.fov_mm.get())
        except ValueError:
            messagebox.showwarning("Check input", "Image ki poori chaudai kitne mm hai wo daalo.",
                                   parent=self)
            return
        if fov <= 0:
            return
        self._set_mmpp(fov / w, f"FOV: {fov} mm across {w} px")

    def _calib_from_roi(self):
        if not self.app.rois:
            messagebox.showwarning("No ROI", "Pehle live view pe ek ROI banao.", parent=self)
            return
        try:
            idx = int(self.roi_pick.get().split("#")[1]) - 1
            dia = float(self.real_dia.get())
        except (ValueError, IndexError):
            messagebox.showwarning("Check input", "ROI chuno aur uska real diameter (mm) daalo.",
                                   parent=self)
            return
        if idx >= len(self.app.rois) or dia <= 0:
            messagebox.showwarning("Check input", "Valid ROI aur diameter > 0 chahiye.", parent=self)
            return
        mmpp = mm_per_px_from_roi(self.app.rois[idx]["r"], dia)
        self._set_mmpp(mmpp, f"ROI #{idx + 1} = {dia} mm")

    def _set_mmpp(self, mmpp, how):
        if mmpp <= 0:
            return
        self.app.mm_per_px.set(mmpp)
        self.app.min_pixels.set(DEFAULT_PARAMS["min_pixels"])   # mm rule takes over
        self.app.calib_pts = []
        self.app.calib_mode = False
        self.app._hide_calib_bar()
        self.app._log(f"calibrated ({how}): {mmpp * 1000:.3f} um/px")
        self._refresh_calib_out()

    def _clear_calib(self):
        self.app.mm_per_px.set(0.0)
        self.app.calib_pts = []
        self.app.calib_mode = False
        self.app._hide_calib_bar()
        self.app._log("calibration cleared - pixel mode")
        self._refresh_calib_out()

    def _set_mm(self):
        try:
            v = float(self.mm_entry.get())
        except ValueError:
            messagebox.showwarning("Check input", "Number daalo (mm).", parent=self)
            return
        if v <= 0:
            return
        self.app.min_length_mm.set(v)
        self.app._log(f"size threshold = {v:.3f} mm")
        self._refresh_calib_out()

    def _set_blur(self, label):
        self.app.blur_ksize.set(0 if label == "off" else int(label.split()[0]))


if __name__ == "__main__":
    App().mainloop()
