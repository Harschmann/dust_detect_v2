"""
camera.py
----------
Enumerates every camera the machine can see and lets the app connect to
a chosen one:

  * Basler industrial cameras via pypylon (each listed by model+serial)
  * USB / UVC webcams and any other OpenCV-openable camera (by index)
  * a synthetic test feed as a last resort so the UI always works

A background thread grabs frames continuously so the UI never blocks.
get_frame() returns a thread-safe copy of the latest frame.
"""

import threading
import time
import numpy as np
import cv2


def enumerate_cameras(max_opencv_index=5):
    """Returns a list of descriptors:
        {"key","kind","name", ...}
    kind is "basler" or "opencv". Always safe to call (never raises)."""
    cams = []

    # --- Basler / pypylon
    try:
        from pypylon import pylon
        for d in pylon.TlFactory.GetInstance().EnumerateDevices():
            serial = d.GetSerialNumber()
            model = d.GetModelName()
            cams.append({
                "key": f"basler:{serial}",
                "kind": "basler",
                "serial": serial,
                "name": f"Basler {model}  ({serial})",
            })
    except Exception:
        pass  # pypylon or runtime not installed - fine

    # --- OpenCV-openable cameras (webcams, UVC, capture cards)
    # Probing missing indices makes OpenCV print to stderr; silence it.
    import os as _os
    import contextlib as _cl
    devnull = _os.open(_os.devnull, _os.O_WRONLY)
    saved_err = _os.dup(2)
    try:
        _os.dup2(devnull, 2)
        for idx in range(max_opencv_index + 1):
            cap = None
            try:
                cap = cv2.VideoCapture(idx)
                if cap is not None and cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        cams.append({
                            "key": f"opencv:{idx}",
                            "kind": "opencv",
                            "index": idx,
                            "name": f"Camera {idx}  (USB / UVC)",
                        })
            except Exception:
                pass
            finally:
                if cap is not None:
                    cap.release()
    finally:
        _os.dup2(saved_err, 2)
        _os.close(devnull)
        _os.close(saved_err)

    return cams


class CameraManager:
    def __init__(self, synthetic_size=(1280, 960)):
        self.synthetic_w, self.synthetic_h = synthetic_size
        self.active = None            # descriptor of the connected camera, or None
        self.mode = "none"            # "basler" | "opencv" | "synthetic" | "none"

        self._pylon = None
        self._camera = None
        self._converter = None
        self._cap = None

        self._lock = threading.Lock()
        self._latest = None
        self._running = False
        self._thread = None

    # ------------------------------------------------------------ connect
    def connect(self, descriptor):
        """descriptor is one item from enumerate_cameras(), or None for the
        synthetic feed. Returns True on success."""
        self._teardown_device()
        ok = False
        if descriptor is None:
            self.mode = "synthetic"
            self.active = None
            ok = True
        elif descriptor["kind"] == "basler":
            ok = self._open_basler(descriptor.get("serial"))
            self.mode = "basler" if ok else "synthetic"
            self.active = descriptor if ok else None
        else:
            ok = self._open_opencv(descriptor["index"])
            self.mode = "opencv" if ok else "synthetic"
            self.active = descriptor if ok else None

        if not self._running:
            self.start()
        return ok

    def connect_best_available(self):
        cams = enumerate_cameras()
        # prefer a Basler if present, else first camera, else synthetic
        basler = next((c for c in cams if c["kind"] == "basler"), None)
        target = basler or (cams[0] if cams else None)
        self.connect(target)
        return cams

    def _open_basler(self, serial):
        try:
            from pypylon import pylon
            self._pylon = pylon
            tlf = pylon.TlFactory.GetInstance()
            device = None
            for d in tlf.EnumerateDevices():
                if serial is None or d.GetSerialNumber() == serial:
                    device = d
                    break
            if device is None:
                return False
            self._camera = pylon.InstantCamera(tlf.CreateDevice(device))
            self._camera.Open()
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            return True
        except Exception as e:
            print(f"[camera] Basler open failed: {e}")
            return False

    def _open_opencv(self, index):
        try:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                return False
            self._cap = cap
            return True
        except Exception as e:
            print(f"[camera] OpenCV open failed: {e}")
            return False

    # ------------------------------------------------------------- thread
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._teardown_device()

    def _teardown_device(self):
        try:
            if self._camera is not None:
                self._camera.StopGrabbing()
                self._camera.Close()
        except Exception:
            pass
        self._camera = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _loop(self):
        t0 = time.time()
        while self._running:
            frame = None
            try:
                if self.mode == "basler":
                    frame = self._grab_basler()
                elif self.mode == "opencv" and self._cap is not None:
                    ok, f = self._cap.read()
                    frame = f if ok else None
                else:
                    frame = self._synthetic(t0)
            except Exception as e:
                print(f"[camera] grab error: {e}")
            if frame is not None:
                with self._lock:
                    self._latest = frame
            time.sleep(0.01)

    def _grab_basler(self):
        grab = self._camera.RetrieveResult(1000, self._pylon.TimeoutHandling_Return)
        try:
            if grab.GrabSucceeded():
                return self._converter.Convert(grab).GetArray()
            return None
        finally:
            grab.Release()

    def _synthetic(self, t0):
        h, w = self.synthetic_h, self.synthetic_w
        frame = np.full((h, w, 3), 24, np.uint8)
        cx, cy = w // 2, h // 2
        cv2.circle(frame, (cx, cy), min(h, w) // 3, (78, 78, 82), -1)
        t = time.time() - t0
        for i in range(3):
            bx = int(cx + 60 * np.sin(t * 0.4 + i * 2.1))
            by = int(cy + 60 * np.cos(t * 0.3 + i * 2.1))
            cv2.circle(frame, (bx, by), 3 + i, (235, 235, 235), -1)
        cv2.putText(frame, "SYNTHETIC FEED - no camera connected", (24, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 240), 1, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------- access
    def get_frame(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def status_text(self):
        if self.mode == "basler" and self.active:
            return f"\U0001F7E2  {self.active['name']}"
        if self.mode == "opencv" and self.active:
            return f"\U0001F7E2  {self.active['name']}"
        return "\U0001F534  Synthetic feed (no camera)"
