#!/usr/bin/env python3
"""
Detectron2 + VideoPose3D GUI
Detectron2 Pipeline:
  Step 1  Install Detectron2
  Step 2  Demo (single image detect)
  Step 3  Batch Image 2D Keypoint Extract   <-- NEW
  Step 4  Extract 2D Keypoints (video)
  Step 5  Convert Format
  Step 6  Download Pretrained Model
  Step 7  3D Inference & Output

VideoPose3D Pipeline:
  Step 1  Install VideoPose3D (patch files)
  Step 2  Extract 2D Keypoints
  Step 3  Convert Format
  Step 4  Download Pretrained Model
  Step 5  3D Inference & Output
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext
import subprocess, threading, os, sys, shutil
from pathlib import Path

# -------------------------------------------------------
# PATH CONFIG
# -------------------------------------------------------
DEFAULT_DETECTRON2  = os.path.expanduser("~/Documents/Video/detectron2")
DEFAULT_VIDEOPOSE3D = os.path.expanduser("~/Documents/Video/VideoPose3D")
DEFAULT_PYTHON      = sys.executable

D2_STEPS = [
    "Step 1  Install Detectron2",
    "Step 2  Demo (detect image)",
    "Step 3  Batch Image 2D Keypoints",
    "Step 4  Video 2D Keypoints",
]

VP_STEPS = [
    "Step 1  Install VideoPose3D (patch files)",
    "Step 2  Extract 2D Keypoints",
    "Step 3  Convert Format",
    "Step 4  Download Pretrained Model",
    "Step 5  3D Inference & Output",
]

C = {
    "bg":        "#0f0f14",
    "panel":     "#1a1a24",
    "panel2":    "#161620",
    "border":    "#2a2a3a",
    "accent":    "#7c6af7",
    "accent2":   "#4ecdc4",
    "success":   "#2ecc71",
    "error":     "#e74c3c",
    "warn":      "#f39c12",
    "text":      "#e8e8f0",
    "muted":     "#6b6b8a",
    "idle":      "#3a3a50",
    "running":   "#f39c12",
    "done":      "#2ecc71",
}

MONO = ("Monospace", 9)
MONO_B = ("Monospace", 9, "bold")
MONO_L = ("Monospace", 10, "bold")
MONO_XL = ("Monospace", 12, "bold")


# =====================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pose3D Pipeline  |  Detectron2 + VideoPose3D")
        self.geometry("1280x820")
        self.resizable(True, True)
        self.configure(bg=C["bg"])

        # paths
        self.d2_path  = tk.StringVar(value=DEFAULT_DETECTRON2)
        self.vp_path  = tk.StringVar(value=DEFAULT_VIDEOPOSE3D)
        self.py_path  = tk.StringVar(value=DEFAULT_PYTHON)

        # D2 inputs
        self.d2_images   = []          # list of image paths (batch)

        # VP inputs
        self.vp_video    = tk.StringVar()
        self.vp_output   = tk.StringVar(value="output_3d.mp4")

        self.running = False
        self._proc   = None

        # step label widgets
        self.d2_step_labels  = []
        self.vp_step_labels  = []

        # step selection checkboxes (for Run All)
        self.d2_step_checks  = [tk.BooleanVar(value=True) for _ in D2_STEPS]
        self.vp_step_checks  = [tk.BooleanVar(value=True) for _ in VP_STEPS]

        self._build_ui()

    # ===================================================
    # UI BUILD
    # ===================================================
    def _build_ui(self):
        # Left: control panel
        left = tk.Frame(self, bg=C["panel"], width=380)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        self._build_header(left)
        self._build_paths(left)

        # Notebook-style tabs
        tab_bar = tk.Frame(left, bg=C["panel"])
        tab_bar.pack(fill="x", padx=0)

        self.tab_d2_frame = tk.Frame(left, bg=C["panel"])
        self.tab_vp_frame = tk.Frame(left, bg=C["panel"])

        self.btn_tab_d2 = tk.Button(tab_bar, text="Detectron2",
            bg=C["accent"], fg="white", relief="flat", font=MONO_B,
            command=self._show_d2_tab)
        self.btn_tab_d2.pack(side="left", fill="x", expand=True, ipady=5)

        self.btn_tab_vp = tk.Button(tab_bar, text="VideoPose3D",
            bg=C["idle"], fg=C["muted"], relief="flat", font=MONO_B,
            command=self._show_vp_tab)
        self.btn_tab_vp.pack(side="left", fill="x", expand=True, ipady=5)

        self._build_d2_tab(self.tab_d2_frame)
        self._build_vp_tab(self.tab_vp_frame)
        self._show_d2_tab()

        # Right: log
        right = tk.Frame(self, bg=C["bg"])
        right.pack(side="right", fill="both", expand=True)
        self._build_log(right)

    def _show_d2_tab(self):
        self.tab_vp_frame.pack_forget()
        self.tab_d2_frame.pack(fill="both", expand=True)
        self.btn_tab_d2.config(bg=C["accent"], fg="white")
        self.btn_tab_vp.config(bg=C["idle"],   fg=C["muted"])

    def _show_vp_tab(self):
        self.tab_d2_frame.pack_forget()
        self.tab_vp_frame.pack(fill="both", expand=True)
        self.btn_tab_vp.config(bg=C["accent2"], fg="#000")
        self.btn_tab_d2.config(bg=C["idle"],    fg=C["muted"])

    # ---------------------------------------------------
    def _lbl(self, parent, text, font=None, color=None):
        return tk.Label(parent, text=text, bg=parent.cget("bg"),
                        fg=color or C["text"], font=font or MONO, anchor="w")

    def _sep(self, parent):
        tk.Frame(parent, bg=C["border"], height=1).pack(fill="x", pady=8)

    def _entry_row(self, parent, label, var, browse_file=False, browse_dir=False):
        row = tk.Frame(parent, bg=parent.cget("bg"))
        row.pack(fill="x", padx=14, pady=2)
        self._lbl(row, label, color=C["muted"]).pack(anchor="w")
        inner = tk.Frame(row, bg=parent.cget("bg"))
        inner.pack(fill="x")
        tk.Entry(inner, textvariable=var, bg=C["border"], fg=C["text"],
                 insertbackground=C["text"], relief="flat",
                 font=MONO).pack(side="left", fill="x", expand=True, ipady=4, padx=(0,3))
        if browse_file:
            tk.Button(inner, text="...", bg=C["accent"], fg="white",
                      relief="flat", font=MONO,
                      command=lambda: var.set(filedialog.askopenfilename())).pack(side="right")
        if browse_dir:
            tk.Button(inner, text="...", bg=C["accent"], fg="white",
                      relief="flat", font=MONO,
                      command=lambda: var.set(filedialog.askdirectory())).pack(side="right")

    def _btn(self, parent, text, cmd, bg=None, fg="white", pady=4):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg or C["idle"], fg=fg, relief="flat",
                         font=MONO, pady=pady)

    # ---------------------------------------------------
    def _build_header(self, parent):
        tk.Frame(parent, bg=C["accent"], height=4).pack(fill="x")
        tk.Frame(parent, bg=parent.cget("bg"), height=8).pack()
        self._lbl(parent, "  POSE3D PIPELINE", font=MONO_XL,
                  color=C["accent"]).pack(anchor="w", padx=14)
        self._lbl(parent, "  Detectron2  >  VideoPose3D", color=C["muted"]).pack(anchor="w", padx=14)
        self._sep(parent)

    def _build_paths(self, parent):
        self._lbl(parent, "  SHARED PATHS", font=MONO_B, color=C["muted"]).pack(anchor="w", padx=14)
        self._entry_row(parent, "Detectron2 root dir",  self.d2_path, browse_dir=True)
        self._entry_row(parent, "VideoPose3D root dir", self.vp_path, browse_dir=True)
        self._entry_row(parent, "Python executable",    self.py_path, browse_file=True)
        self._sep(parent)

    # ---------------------------------------------------
    # DETECTRON2 TAB
    # ---------------------------------------------------
    def _build_d2_tab(self, parent):
        # Input/Output  ── images only (2D pipeline)
        self._lbl(parent, "  INPUT / OUTPUT", font=MONO_B, color=C["muted"]).pack(anchor="w", padx=14, pady=(6,0))

        # Batch images selector
        batch_row = tk.Frame(parent, bg=parent.cget("bg"))
        batch_row.pack(fill="x", padx=14, pady=4)
        self._lbl(batch_row, "Batch images (Step 3)", color=C["muted"]).pack(anchor="w")
        btn_row = tk.Frame(batch_row, bg=parent.cget("bg"))
        btn_row.pack(fill="x")
        self._btn(btn_row, "Select Images...", self._select_batch_images,
                  bg=C["accent"]).pack(side="left", padx=(0,4))
        self._btn(btn_row, "Clear", self._clear_batch_images,
                  bg=C["idle"]).pack(side="left")
        self.batch_count_lbl = self._lbl(batch_row, "No images selected.", color=C["muted"])
        self.batch_count_lbl.pack(anchor="w", pady=(2,0))

        # Video input for Step 4 (Video 2D Keypoints) — shared with VP pipeline
        self._lbl(parent, "  Video input (Step 4) — also used by VideoPose3D",
                  font=MONO_B, color=C["muted"]).pack(anchor="w", padx=14, pady=(8,0))
        self._entry_row(parent, "Input video (.mp4)", self.vp_video, browse_file=True)

        self._sep(parent)

        # Steps display with checkboxes
        self._lbl(parent, "  PIPELINE STEPS", font=MONO_B, color=C["muted"]).pack(anchor="w", padx=14)
        for i, name in enumerate(D2_STEPS):
            row = tk.Frame(parent, bg=parent.cget("bg"))
            row.pack(fill="x", padx=14, pady=1)
            # checkbox for Run All selection
            cb = tk.Checkbutton(row, variable=self.d2_step_checks[i],
                                bg=parent.cget("bg"), fg=C["muted"],
                                selectcolor=C["border"], activebackground=parent.cget("bg"),
                                relief="flat", bd=0)
            cb.pack(side="left")
            dot = tk.Label(row, text="*", bg=parent.cget("bg"), fg=C["idle"], font=MONO_B)
            dot.pack(side="left")
            self._lbl(row, f"  {name}").pack(side="left")
            self.d2_step_labels.append(dot)

        self._sep(parent)

        # Run all
        self._btn(parent, "[>] RUN ALL (Detectron2)", self._d2_run_all,
                  bg=C["accent"], fg="white", pady=8).pack(fill="x", padx=14)

        self._btn(parent, "[x] STOP", self._stop,
                  bg=C["error"], fg="white").pack(fill="x", padx=14, pady=(6,2))

    def _select_batch_images(self):
        files = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All", "*.*")])
        if files:
            self.d2_images = list(files)
            self.batch_count_lbl.config(
                text=f"{len(self.d2_images)} image(s) selected.", fg=C["success"])

    def _clear_batch_images(self):
        self.d2_images = []
        self.batch_count_lbl.config(text="No images selected.", fg=C["muted"])

    # ---------------------------------------------------
    # VIDEOPOSE3D TAB
    # ---------------------------------------------------
    def _build_vp_tab(self, parent):
        self._lbl(parent, "  INPUT / OUTPUT", font=MONO_B, color=C["muted"]).pack(anchor="w", padx=14, pady=(6,0))
        self._entry_row(parent, "Input video (.mp4)", self.vp_video,  browse_file=True)
        self._entry_row(parent, "Output file",        self.vp_output)
        self._sep(parent)

        self._lbl(parent, "  PIPELINE STEPS", font=MONO_B, color=C["muted"]).pack(anchor="w", padx=14)
        for i, name in enumerate(VP_STEPS):
            row = tk.Frame(parent, bg=parent.cget("bg"))
            row.pack(fill="x", padx=14, pady=1)
            cb = tk.Checkbutton(row, variable=self.vp_step_checks[i],
                                bg=parent.cget("bg"), fg=C["muted"],
                                selectcolor=C["border"], activebackground=parent.cget("bg"),
                                relief="flat", bd=0)
            cb.pack(side="left")
            dot = tk.Label(row, text="*", bg=parent.cget("bg"), fg=C["idle"], font=MONO_B)
            dot.pack(side="left")
            self._lbl(row, f"  {name}").pack(side="left")
            self.vp_step_labels.append(dot)

        self._sep(parent)

        self._btn(parent, "[>] RUN ALL (VideoPose3D)", self._vp_run_all,
                  bg=C["accent2"], fg="#000", pady=8).pack(fill="x", padx=14)

        self._btn(parent, "[x] STOP", self._stop,
                  bg=C["error"], fg="white").pack(fill="x", padx=14, pady=(6,2))

    # ---------------------------------------------------
    # LOG
    # ---------------------------------------------------
    def _build_log(self, parent):
        hdr = tk.Frame(parent, bg=C["panel"])
        hdr.pack(fill="x")
        self._lbl(hdr, "  TERMINAL OUTPUT", font=MONO_L,
                  color=C["accent2"]).pack(side="left", padx=14, pady=6)
        self._btn(hdr, "Clear", self._clear_log, bg=C["border"]).pack(side="right", padx=14, pady=4)

        self.log = scrolledtext.ScrolledText(
            parent, bg="#080810", fg=C["text"], font=MONO,
            relief="flat", insertbackground=C["text"], wrap="word")
        self.log.pack(fill="both", expand=True)

        self.log.tag_config("info",    foreground=C["accent2"])
        self.log.tag_config("success", foreground=C["success"])
        self.log.tag_config("error",   foreground=C["error"])
        self.log.tag_config("warn",    foreground=C["warn"])
        self.log.tag_config("cmd",     foreground=C["accent"])
        self.log.tag_config("step",    foreground="#fff",
                            background=C["accent"], font=MONO_B)
        self.log.tag_config("step2",   foreground="#000",
                            background=C["accent2"], font=MONO_B)

    def _log(self, text, tag=""):
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.update_idletasks()

    def _clear_log(self):
        self.log.delete("1.0", "end")

    # ---------------------------------------------------
    # Step indicator
    # ---------------------------------------------------
    def _set_d2_step(self, idx, state):
        if idx < len(self.d2_step_labels):
            colors = {"idle": C["idle"], "running": C["running"],
                      "done": C["done"], "error": C["error"]}
            self.d2_step_labels[idx].config(fg=colors.get(state, C["muted"]))

    def _set_vp_step(self, idx, state):
        if idx < len(self.vp_step_labels):
            colors = {"idle": C["idle"], "running": C["running"],
                      "done": C["done"], "error": C["error"]}
            self.vp_step_labels[idx].config(fg=colors.get(state, C["muted"]))

    # ---------------------------------------------------
    # Command runner
    # ---------------------------------------------------
    def _run_cmd(self, cmd, cwd=None, step_fn=None, state_idx=None):
        """Run shell command, stream output to log. Returns True/False."""
        self._log(f"\n$ {' '.join(str(c) for c in cmd)}", "cmd")
        if step_fn and state_idx is not None:
            step_fn(state_idx, "running")
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=cwd, text=True, bufsize=1,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            self._proc = proc
            for line in proc.stdout:
                low = line.lower()
                tag = ""
                if "error" in low or "traceback" in low or "failed" in low:
                    tag = "error"
                elif "warning" in low or "warn" in low:
                    tag = "warn"
                elif "done" in low or "success" in low or "saved" in low:
                    tag = "success"
                self._log(line.rstrip(), tag)
            proc.wait()
            ok = proc.returncode == 0
            if step_fn and state_idx is not None:
                step_fn(state_idx, "done" if ok else "error")
            return ok
        except Exception as e:
            self._log(f"Command failed: {e}", "error")
            if step_fn and state_idx is not None:
                step_fn(state_idx, "error")
            return False

    def _py(self): return self.py_path.get() or sys.executable
    def _d2(self):  return self.d2_path.get()
    def _vp(self):  return self.vp_path.get()

    # ===================================================
    # DETECTRON2 STEPS
    # ===================================================
    def _d2_step_install(self):
        self._log("\n=== D2 Step 1: Install Detectron2 ===", "step")
        d2 = self._d2()
        if not os.path.isdir(d2):
            self._log("Detectron2 directory not found. Please git clone first.", "error")
            self._set_d2_step(0, "error"); return False
        ok = self._run_cmd(
            [self._py(), "-m", "pip", "install", "-e", ".", "--no-build-isolation"],
            cwd=d2, step_fn=self._set_d2_step, state_idx=0)
        if ok: self._log("[OK] Detectron2 installed", "success")
        return ok

    def _d2_step_demo(self):
        self._log("\n=== D2 Step 2: Detectron2 Demo ===", "step")
        d2 = self._d2()
        demo_img = os.path.join(d2, "demo", "input.jpg")
        out_dir  = os.path.join(d2, "demo", "output")
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isfile(demo_img):
            self._log("demo/input.jpg not found, downloading test image...", "warn")
            ok = self._run_cmd(["wget", "-q", "-O", demo_img,
                "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/"
                "Shopping_Centre_with_car_park.jpg/640px-Shopping_Centre_with_car_park.jpg"])
            if not ok:
                self._log("Download failed. Put an image at demo/input.jpg manually.", "error")
                self._set_d2_step(1, "error"); return False

        self._fix_demo_import(d2)

        ok = self._run_cmd([
            self._py(), "demo/demo.py",
            "--config-file", "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            "--input", "demo/input.jpg",
            "--output", "demo/output/",
            "--opts", "MODEL.WEIGHTS",
            "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"],
            cwd=d2, step_fn=self._set_d2_step, state_idx=1)
        if ok: self._log(f"[OK] Result saved to {out_dir}", "success")
        return ok

    def _d2_step_batch_images(self):
        self._log("\n=== D2 Step 3: Batch Image 2D Keypoints ===", "step")
        d2 = self._d2()
        if not self.d2_images:
            self._log("No images selected. Click 'Select Images...' first.", "error")
            self._set_d2_step(2, "error"); return False

        self._fix_demo_import(d2)

        # Use detectron2's own model_zoo API to resolve + cache the model locally.
        # This uses detectron2's PathManager/S3 handler which follows redirects correctly,
        # rather than calling wget directly on the S3 URL (which returns 403).
        self._log("Resolving keypoint model via detectron2 model_zoo (downloads on first run)...", "info")
        resolve_script = (
            "import sys, os; "
            "sys.path.insert(0, os.path.join('" + d2.replace("'", "\\'") + "')); "
            "from detectron2 import model_zoo; "
            "url = model_zoo.get_checkpoint_url("
            "'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'); "
            "from detectron2.utils.file_io import PathManager; "
            "local = PathManager.get_local_path(url); "
            "print('MODEL_PATH:' + local)"
        )
        import subprocess as _sp
        try:
            result = _sp.run(
                [self._py(), "-c", resolve_script],
                capture_output=True, text=True,
                cwd=d2, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            model_local = None
            for line in (result.stdout + result.stderr).splitlines():
                self._log(line, "")
                if line.startswith("MODEL_PATH:"):
                    model_local = line[len("MODEL_PATH:"):].strip()
            if not model_local or not os.path.isfile(model_local):
                self._log("[ERROR] Could not resolve model path via model_zoo.", "error")
                self._log("  Hint: Run 'Step 1 Install Detectron2' first, then retry.", "warn")
                self._set_d2_step(2, "error"); return False
            self._log(f"[OK] Using model: {model_local}", "success")
        except Exception as e:
            self._log(f"[ERROR] model_zoo resolve failed: {e}", "error")
            self._set_d2_step(2, "error"); return False

        out_dir = os.path.join(d2, "demo", "batch_output")
        os.makedirs(out_dir, exist_ok=True)
        self._log(f"Processing {len(self.d2_images)} image(s)...", "info")

        all_ok = True
        for i, img_path in enumerate(self.d2_images):
            if not self.running:
                self._log("[STOPPED] Batch stopped by user.", "warn")
                break
            self._log(f"  [{i+1}/{len(self.d2_images)}] {Path(img_path).name}", "info")
            ok = self._run_cmd([
                self._py(), "demo/demo.py",
                "--config-file", "configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
                "--input", img_path,
                "--output", out_dir,
                "--opts", "MODEL.WEIGHTS", model_local],
                cwd=d2)
            if not ok:
                self._log(f"  [WARN] Failed on {Path(img_path).name}", "warn")
                all_ok = False

        self._set_d2_step(2, "done" if all_ok else "error")
        if all_ok:
            self._log(f"[OK] Batch keypoint detection done. Results in {out_dir}", "success")
        else:
            self._log("[WARN] Some images failed. Check log above.", "warn")
        return all_ok

    def _d2_step_infer_video(self):
        self._log("\n=== D2 Step 4: Video 2D Keypoints ===", "step")
        vp = self._vp()
        video = self.vp_video.get()
        if not video or not os.path.isfile(video):
            self._log("Please select an input video first!", "error")
            self._set_d2_step(3, "error"); return False

        video_dir = os.path.join(vp, "my_videos")
        os.makedirs(video_dir, exist_ok=True)
        dst = os.path.join(video_dir, Path(video).name)
        if os.path.abspath(video) != os.path.abspath(dst):
            shutil.copy2(video, dst)
            self._log(f"Video copied to {dst}", "info")

        os.makedirs(os.path.join(vp, "npz_output"), exist_ok=True)
        self._fix_infer_numpy(vp)

        ok = self._run_cmd([
            self._py(), "inference/infer_video_d2.py",
            "--cfg", "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
            "--output-dir", "npz_output",
            "--image-ext", "mp4",
            "my_videos/"],
            cwd=vp, step_fn=self._set_d2_step, state_idx=3)
        if ok: self._log("[OK] 2D keypoints saved to npz_output/", "success")
        return ok

    def _d2_run_all(self):
        checked = [i for i, v in enumerate(self.d2_step_checks) if v.get()]
        # Step index 2 = batch images, step index 3 = video 2D
        if 2 in checked and not self.d2_images:
            self._log("[ERROR] Step 3 (Batch Images) is checked but no images selected!", "error")
            return
        if 3 in checked and not self.vp_video.get():
            self._log("[ERROR] Step 4 (Video 2D) requires an input video but none is selected!", "error")
            return

        step_fns = [
            self._d2_step_install,
            self._d2_step_demo,
            self._d2_step_batch_images,
            self._d2_step_infer_video,
        ]
        def worker():
            for i, fn in enumerate(step_fns):
                if i not in checked:
                    self._log(f"\n[SKIP] D2 {D2_STEPS[i]} (unchecked)", "warn")
                    continue
                if not self.running: self._log("\n[STOPPED]", "warn"); break
                if not fn():
                    self._log("\n[ERROR] Pipeline stopped.", "error"); break
            else:
                self._log("\n[DONE] All selected Detectron2 steps completed!", "success")
            self._finish_run()
        self._start_run(worker)

    # ===================================================
    # VIDEOPOSE3D STEPS
    # ===================================================
    def _vp_step_install(self):
        self._log("\n=== VP Step 1: Install VideoPose3D + Patch Files ===", "step2")
        vp = self._vp()
        if not os.path.isdir(vp):
            self._log("VideoPose3D directory not found. Please git clone first.", "error")
            self._set_vp_step(0, "error"); return False

        # Patch 1: infer_video_d2.py - numpy dtype fix
        self._fix_infer_numpy(vp)

        # Patch 2: visualization.py - fps=None fix
        self._fix_viz_fps(vp)

        # Patch 3: demo.py import fix (in detectron2)
        d2 = self._d2()
        if os.path.isdir(d2):
            self._fix_demo_import(d2)

        # Install dependencies
        req_file = os.path.join(vp, "requirements.txt")
        if os.path.isfile(req_file):
            self._log("Installing requirements.txt...", "info")
            ok = self._run_cmd(
                [self._py(), "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=vp)
        else:
            self._log("No requirements.txt found, installing matplotlib...", "warn")
            ok = self._run_cmd(
                [self._py(), "-m", "pip", "install", "matplotlib"])

        self._set_vp_step(0, "done" if ok else "error")
        if ok:
            self._log("[OK] VideoPose3D installed and all files patched", "success")
            self._log("  Patched: inference/infer_video_d2.py (numpy dtype)", "info")
            self._log("  Patched: common/visualization.py (fps=None)", "info")
            self._log("  Patched: detectron2/demo/demo.py (import path)", "info")
        return ok

    def _vp_step_infer(self):
        self._log("\n=== VP Step 2: Extract 2D Keypoints ===", "step2")
        vp = self._vp()
        video = self.vp_video.get()
        if not video or not os.path.isfile(video):
            self._log("Please select an input video first!", "error")
            self._set_vp_step(1, "error"); return False

        video_dir = os.path.join(vp, "my_videos")
        os.makedirs(video_dir, exist_ok=True)
        dst = os.path.join(video_dir, Path(video).name)
        if os.path.abspath(video) != os.path.abspath(dst):
            shutil.copy2(video, dst)
            self._log(f"Video copied to {dst}", "info")

        os.makedirs(os.path.join(vp, "npz_output"), exist_ok=True)
        self._fix_infer_numpy(vp)

        ok = self._run_cmd([
            self._py(), "inference/infer_video_d2.py",
            "--cfg", "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
            "--output-dir", "npz_output",
            "--image-ext", "mp4",
            "my_videos/"],
            cwd=vp, step_fn=self._set_vp_step, state_idx=1)
        if ok: self._log("[OK] 2D keypoints saved to npz_output/", "success")
        return ok

    def _vp_step_prepare(self):
        self._log("\n=== VP Step 3: Convert Format ===", "step2")
        vp = self._vp()
        ok = self._run_cmd([
            self._py(), "prepare_data_2d_custom.py",
            "-i", "../npz_output", "-o", "myvideos"],
            cwd=os.path.join(vp, "data"),
            step_fn=self._set_vp_step, state_idx=2)
        if ok: self._log("[OK] data/data_2d_custom_myvideos.npz created", "success")
        return ok

    def _vp_step_download(self):
        self._log("\n=== VP Step 4: Download Pretrained Model ===", "step2")
        vp = self._vp()
        ckpt_dir = os.path.join(vp, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        model = os.path.join(ckpt_dir, "pretrained_h36m_cpn.bin")
        if os.path.isfile(model):
            self._log("Model already exists, skipping download.", "warn")
            self._set_vp_step(3, "done"); return True
        ok = self._run_cmd([
            "wget", "--show-progress",
            "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin",
            "-P", ckpt_dir],
            step_fn=self._set_vp_step, state_idx=3)
        if ok: self._log("[OK] Pretrained model downloaded", "success")
        return ok

    def _vp_step_run3d(self):
        self._log("\n=== VP Step 5: 3D Inference & Output ===", "step2")
        vp = self._vp()
        video = self.vp_video.get()
        if not video:
            self._log("Please select an input video first!", "error")
            self._set_vp_step(4, "error"); return False

        self._fix_viz_fps(vp)

        subject = Path(video).name
        output  = self.vp_output.get() or "output_3d.mp4"
        ok = self._run_cmd([
            self._py(), "run.py",
            "-d", "custom", "-k", "myvideos", "-arc", "3,3,3,3,3",
            "-c", "checkpoint", "--evaluate", "pretrained_h36m_cpn.bin",
            "--render",
            "--viz-subject", subject, "--viz-action", "custom",
            "--viz-camera", "0", "--viz-output", output,
            "--viz-size", "5", "--viz-downsample", "2"],
            cwd=vp, step_fn=self._set_vp_step, state_idx=4)
        if ok: self._log(f"[OK] Output saved: {os.path.join(vp, output)}", "success")
        return ok

    def _vp_run_all(self):
        checked = [i for i, v in enumerate(self.vp_step_checks) if v.get()]
        # VP step index 1 = Extract 2D Keypoints, step index 4 = 3D Inference
        if (1 in checked or 4 in checked) and not self.vp_video.get():
            self._log("[ERROR] VP Step 2/5 requires an input video but none is selected!", "error")
            return

        step_fns = [
            self._vp_step_install,
            self._vp_step_infer,
            self._vp_step_prepare,
            self._vp_step_download,
            self._vp_step_run3d,
        ]
        def worker():
            for i, fn in enumerate(step_fns):
                if i not in checked:
                    self._log(f"\n[SKIP] VP {VP_STEPS[i]} (unchecked)", "warn")
                    continue
                if not self.running: self._log("\n[STOPPED]", "warn"); break
                if not fn():
                    self._log("\n[ERROR] Pipeline stopped.", "error"); break
            else:
                self._log("\n[DONE] All selected VideoPose3D steps completed!", "success")
            self._finish_run()
        self._start_run(worker)

    # ===================================================
    # PATCH HELPERS  (shared between D2 and VP pipelines)
    # ===================================================
    def _fix_demo_import(self, d2):
        demo_py = os.path.join(d2, "demo", "demo.py")
        if not os.path.isfile(demo_py): return
        with open(demo_py, "r") as f: src = f.read()
        if "vision.fair" in src:
            src = src.replace(
                "from vision.fair.detectron2.demo.predictor import VisualizationDemo",
                "from predictor import VisualizationDemo")
            with open(demo_py, "w") as f: f.write(src)
            self._log("[FIXED] demo.py: corrected import path (vision.fair -> predictor)", "warn")

    def _fix_infer_numpy(self, vp):
        infer_py = os.path.join(vp, "inference", "infer_video_d2.py")
        if not os.path.isfile(infer_py): return
        with open(infer_py, "r") as f: src = f.read()
        if "dtype=object" not in src:
            src = src.replace(
                "np.savez_compressed(out_name, boxes=boxes, segments=segments, "
                "keypoints=keypoints, metadata=metadata)",
                "np.savez_compressed(out_name, "
                "boxes=np.array(boxes, dtype=object), "
                "segments=np.array(segments, dtype=object), "
                "keypoints=np.array(keypoints, dtype=object), "
                "metadata=metadata)")
            with open(infer_py, "w") as f: f.write(src)
            self._log("[FIXED] infer_video_d2.py: numpy inhomogeneous shape fix applied", "warn")

    def _fix_viz_fps(self, vp):
        viz_py = os.path.join(vp, "common", "visualization.py")
        if not os.path.isfile(viz_py): return
        with open(viz_py, "r") as f: src = f.read()
        if "fps = fps or 30" not in src:
            src = src.replace(
                "fps /= downsample",
                "fps = fps or 30\n    fps /= downsample")
            with open(viz_py, "w") as f: f.write(src)
            self._log("[FIXED] visualization.py: fps=None fallback to 30 applied", "warn")

    # ===================================================
    # RUN CONTROL
    # ===================================================
    def _start_run(self, worker_fn):
        if self.running: return
        self.running = True
        threading.Thread(target=worker_fn, daemon=True).start()

    def _finish_run(self):
        self.running = False

    def _stop(self):
        self.running = False
        if self._proc:
            try: self._proc.terminate()
            except Exception: pass
        self._log("\n[STOPPED] Aborted by user.", "warn")


if __name__ == "__main__":
    app = App()
    app.mainloop()
