#!/usr/bin/env python3
"""
Detectron2 + VideoPose3D GUI  —  Windows Only
============================================
Shared Area:
  Create venv  (shared by d2 and p3d)
  Install packages  (torch CUDA + deps)

Detectron2 Pipeline:
  Step 1  Install Detectron2  (prebuilt wheel via miropsota)
  Step 2  Batch Image 2D Keypoints
  Step 3  Video 2D Keypoints

VideoPose3D Pipeline:
  Step 1  Install VideoPose3D (patch files)
  Step 2  Extract 2D Keypoints
  Step 3  Convert Format
  Step 4  Download Pretrained Model
  Step 5  3D Inference & Output
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext
import subprocess, threading, os, sys, shutil, urllib.request
from pathlib import Path

# -------------------------------------------------------
# PATH CONFIG
# -------------------------------------------------------
DEFAULT_BASE        = str(Path.home() / "Documents" / "d2_p3d")
DEFAULT_DETECTRON2  = str(Path(DEFAULT_BASE) / "detectron2")
DEFAULT_VIDEOPOSE3D = str(Path(DEFAULT_BASE) / "VideoPose3D")
DEFAULT_VENV        = str(Path(DEFAULT_BASE) / "venv")
DEFAULT_PYTHON      = str(Path(DEFAULT_VENV) / "Scripts" / "python.exe")

LOG_MAX_LINES = 3000   # trim old lines beyond this to prevent memory buildup

D2_STEPS = [
    "Step 1  Install Detectron2",
    "Step 2  Batch Image 2D Keypoints",
    "Step 3  Video 2D Keypoints",
]

VP_STEPS = [
    "Step 1  Install VideoPose3D (patch files)",
    "Step 2  Extract 2D Keypoints",
    "Step 3  Convert Format",
    "Step 4  Download Pretrained Model",
    "Step 5  3D Inference & Output",
]

C = {
    "bg":      "#0f0f14",
    "panel":   "#1a1a24",
    "border":  "#2a2a3a",
    "accent":  "#7c6af7",
    "accent2": "#4ecdc4",
    "success": "#2ecc71",
    "error":   "#e74c3c",
    "warn":    "#f39c12",
    "text":    "#e8e8f0",
    "muted":   "#6b6b8a",
    "idle":    "#3a3a50",
    "running": "#f39c12",
    "done":    "#2ecc71",
}

MONO   = ("Monospace", 9)
MONO_B = ("Monospace", 9, "bold")
MONO_L = ("Monospace", 10, "bold")
MONO_XL= ("Monospace", 12, "bold")


# =====================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pose3D Pipeline  |  Detectron2 + VideoPose3D  [Windows]")
        self.geometry("1280x820")
        self.resizable(True, True)
        self.configure(bg=C["bg"])

        # ── paths ──────────────────────────────────────
        self.d2_path   = tk.StringVar(value=DEFAULT_DETECTRON2)
        self.vp_path   = tk.StringVar(value=DEFAULT_VIDEOPOSE3D)
        self.venv_path = tk.StringVar(value=DEFAULT_VENV)
        self.py_path   = tk.StringVar(value=DEFAULT_PYTHON)

        # ── D2 inputs ──────────────────────────────────
        self.d2_images = []
        # CUDA version: cu128/cu126/cu130/cpu
        # miropsota detectron2 wheels: latest is pt2.10.0 for all of these
        self.torch_cuda_ver = tk.StringVar(value="cu128")
        self.d2_wheel_pkg   = tk.StringVar(
            value="detectron2==0.6+fd27788pt2.10.0cu128")

        # ── VP inputs ──────────────────────────────────
        self.vp_video  = tk.StringVar()
        self.vp_output = tk.StringVar(value="output_videos/output_3d.mp4")

        # ── runtime state ──────────────────────────────
        self.running = False
        self._proc   = None

        # ── step label widgets ─────────────────────────
        self.d2_step_labels = []
        self.vp_step_labels = []

        # ── step checkboxes (for Run All) ──────────────
        self.d2_step_checks = [tk.BooleanVar(value=True) for _ in D2_STEPS]
        self.vp_step_checks = [tk.BooleanVar(value=True) for _ in VP_STEPS]

        self._build_ui()

    # ===================================================
    # UI BUILD
    # ===================================================
    def _build_ui(self):
        # ── Scrollable left panel ──────────────────────
        left_outer = tk.Frame(self, bg=C["panel"], width=440)
        left_outer.pack(side="left", fill="y")
        left_outer.pack_propagate(False)

        canvas = tk.Canvas(left_outer, bg=C["panel"], highlightthickness=0)
        scrollbar = tk.Scrollbar(left_outer, orient="vertical",
                                 command=canvas.yview)
        left = tk.Frame(canvas, bg=C["panel"])
        left.bind("<Configure>",
                  lambda e: canvas.configure(
                      scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=left, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(
                            int(-1 * (e.delta / 120)), "units"))

        # ── sections ───────────────────────────────────
        self._build_header(left)
        self._build_paths(left)
        self._build_env_area(left)      # venv + CUDA + install
        self._build_d2_install_mode(left)  # wheel pkg + auto-detect

        # ── tabs ───────────────────────────────────────
        tab_bar = tk.Frame(left, bg=C["panel"])
        tab_bar.pack(fill="x")

        self.tab_d2_frame = tk.Frame(left, bg=C["panel"])
        self.tab_vp_frame = tk.Frame(left, bg=C["panel"])

        self.btn_tab_d2 = tk.Button(
            tab_bar, text="Detectron2",
            bg=C["accent"], fg="white", relief="flat", font=MONO_B,
            command=self._show_d2_tab)
        self.btn_tab_d2.pack(side="left", fill="x", expand=True, ipady=5)

        self.btn_tab_vp = tk.Button(
            tab_bar, text="VideoPose3D",
            bg=C["idle"], fg=C["muted"], relief="flat", font=MONO_B,
            command=self._show_vp_tab)
        self.btn_tab_vp.pack(side="left", fill="x", expand=True, ipady=5)

        self._build_d2_tab(self.tab_d2_frame)
        self._build_vp_tab(self.tab_vp_frame)
        self._show_d2_tab()

        # ── right: log ────────────────────────────────
        right = tk.Frame(self, bg=C["bg"])
        right.pack(side="right", fill="both", expand=True)
        self._build_log(right)

    def _show_d2_tab(self):
        self.tab_vp_frame.pack_forget()
        self.tab_d2_frame.pack(fill="both", expand=True)
        self.btn_tab_d2.config(bg=C["accent"],  fg="white")
        self.btn_tab_vp.config(bg=C["idle"],    fg=C["muted"])

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

    def _entry_row(self, parent, label, var,
                   browse_file=False, browse_dir=False):
        row = tk.Frame(parent, bg=parent.cget("bg"))
        row.pack(fill="x", padx=14, pady=2)
        self._lbl(row, label, color=C["muted"]).pack(anchor="w")
        inner = tk.Frame(row, bg=parent.cget("bg"))
        inner.pack(fill="x")
        tk.Entry(inner, textvariable=var, bg=C["border"], fg=C["text"],
                 insertbackground=C["text"], relief="flat",
                 font=MONO).pack(side="left", fill="x", expand=True,
                                 ipady=4, padx=(0, 3))
        if browse_file:
            tk.Button(inner, text="...", bg=C["accent"], fg="white",
                      relief="flat", font=MONO,
                      command=lambda: var.set(
                          filedialog.askopenfilename())).pack(side="right")
        if browse_dir:
            tk.Button(inner, text="...", bg=C["accent"], fg="white",
                      relief="flat", font=MONO,
                      command=lambda: var.set(
                          filedialog.askdirectory())).pack(side="right")

    def _btn(self, parent, text, cmd, bg=None, fg="white", pady=4):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg or C["idle"], fg=fg, relief="flat",
                         font=MONO, pady=pady)

    # ---------------------------------------------------
    def _build_header(self, parent):
        tk.Frame(parent, bg=C["accent"], height=4).pack(fill="x")
        tk.Frame(parent, bg=parent.cget("bg"), height=8).pack()
        self._lbl(parent, "  POSE3D PIPELINE  [Windows]",
                  font=MONO_XL, color=C["accent"]).pack(anchor="w", padx=14)
        self._lbl(parent, "  Detectron2  >  VideoPose3D",
                  color=C["muted"]).pack(anchor="w", padx=14)
        self._sep(parent)

    def _build_paths(self, parent):
        self._lbl(parent, "  SHARED PATHS",
                  font=MONO_B, color=C["muted"]).pack(anchor="w", padx=14)
        self._entry_row(parent, "Detectron2 root dir",
                        self.d2_path,   browse_dir=True)
        self._entry_row(parent, "VideoPose3D root dir",
                        self.vp_path,   browse_dir=True)
        self._entry_row(parent, "Venv dir (shared)",
                        self.venv_path, browse_dir=True)
        self._sep(parent)

    # ---------------------------------------------------
    # ENVIRONMENT SETUP (venv + CUDA + install)
    # ---------------------------------------------------
    def _build_env_area(self, parent):
        self._lbl(parent, "  ENVIRONMENT SETUP",
                  font=MONO_B, color=C["muted"]).pack(anchor="w", padx=14)

        # CUDA version selector
        self._lbl(parent,
                  "  PyTorch CUDA version — must match your GPU driver:",
                  color=C["muted"]).pack(anchor="w", padx=14)
        cuda_row = tk.Frame(parent, bg=parent.cget("bg"))
        cuda_row.pack(fill="x", padx=14, pady=(2, 6))
        for label, val in [
            ("CUDA 12.8  (cu128, recommended)", "cu128"),
            ("CUDA 12.6  (cu126)",              "cu126"),
            ("CUDA 13.0  (cu130)",              "cu130"),
            ("CPU only   (no GPU)",             "cpu"),
        ]:
            tk.Radiobutton(
                cuda_row, text=label,
                variable=self.torch_cuda_ver, value=val,
                bg=parent.cget("bg"), fg=C["text"],
                selectcolor=C["border"],
                activebackground=parent.cget("bg"),
                font=MONO, relief="flat",
                command=self._sync_d2_pkg_from_cuda,
            ).pack(anchor="w")

        # Buttons
        btn_row = tk.Frame(parent, bg=parent.cget("bg"))
        btn_row.pack(fill="x", padx=14, pady=4)
        self._btn(btn_row, "[+] Create Venv",
                  self._create_venv,
                  bg=C["accent"]).pack(side="left", padx=(0, 4))
        self._btn(btn_row, "[^] Install Packages",
                  self._install_packages,
                  bg=C["accent2"], fg="#000").pack(side="left", padx=(0, 4))
        self._btn(btn_row, "[x] STOP",
                  self._stop,
                  bg=C["error"]).pack(side="left")

        # Python path display
        py_row = tk.Frame(parent, bg=parent.cget("bg"))
        py_row.pack(fill="x", padx=14, pady=(2, 0))
        self._lbl(py_row, "Python in venv",
                  color=C["muted"]).pack(anchor="w")
        inner = tk.Frame(py_row, bg=parent.cget("bg"))
        inner.pack(fill="x")
        tk.Entry(inner, textvariable=self.py_path,
                 bg=C["border"], fg=C["text"],
                 insertbackground=C["text"], relief="flat",
                 font=MONO).pack(side="left", fill="x", expand=True,
                                 ipady=4, padx=(0, 3))
        tk.Button(inner, text="...", bg=C["accent"], fg="white",
                  relief="flat", font=MONO,
                  command=lambda: self.py_path.set(
                      filedialog.askopenfilename())).pack(side="right")
        self._sep(parent)

    # ---------------------------------------------------
    # D2 INSTALL MODE (wheel only on Windows)
    # ---------------------------------------------------
    def _build_d2_install_mode(self, parent):
        self._lbl(parent, "  DETECTRON2 INSTALL",
                  font=MONO_B, color=C["warn"]).pack(anchor="w", padx=14)
        self._lbl(parent,
                  "  Uses miropsota community wheel (pt2.10.0).",
                  color=C["muted"]).pack(anchor="w", padx=14)
        self._lbl(parent,
                  "  Browse: miropsota.github.io/torch_packages_builder/detectron2",
                  color=C["muted"]).pack(anchor="w", padx=14)

        pkg_row = tk.Frame(parent, bg=parent.cget("bg"))
        pkg_row.pack(fill="x", padx=14, pady=(4, 0))
        tk.Entry(pkg_row, textvariable=self.d2_wheel_pkg,
                 bg=C["border"], fg=C["text"],
                 insertbackground=C["text"], relief="flat",
                 font=MONO).pack(side="left", fill="x", expand=True,
                                 ipady=4, padx=(0, 4))
        self._btn(pkg_row, "Auto-detect",
                  self._auto_detect_d2_pkg,
                  bg=C["accent"]).pack(side="right")
        self._sep(parent)

    # ---------------------------------------------------
    # DETECTRON2 TAB
    # ---------------------------------------------------
    def _build_d2_tab(self, parent):
        self._lbl(parent, "  INPUT / OUTPUT",
                  font=MONO_B, color=C["muted"]).pack(
                      anchor="w", padx=14, pady=(6, 0))

        # Batch images
        batch_row = tk.Frame(parent, bg=parent.cget("bg"))
        batch_row.pack(fill="x", padx=14, pady=4)
        self._lbl(batch_row, "Batch images (Step 2)",
                  color=C["muted"]).pack(anchor="w")
        btn_row = tk.Frame(batch_row, bg=parent.cget("bg"))
        btn_row.pack(fill="x")
        self._btn(btn_row, "Select Images...",
                  self._select_batch_images,
                  bg=C["accent"]).pack(side="left", padx=(0, 4))
        self._btn(btn_row, "Clear",
                  self._clear_batch_images,
                  bg=C["idle"]).pack(side="left")
        self.batch_count_lbl = self._lbl(
            batch_row, "No images selected.", color=C["muted"])
        self.batch_count_lbl.pack(anchor="w", pady=(2, 0))

        # Video input
        self._lbl(parent,
                  "  Video input (Step 3) — also used by VideoPose3D",
                  font=MONO_B, color=C["muted"]).pack(
                      anchor="w", padx=14, pady=(8, 0))
        self._entry_row(parent, "Input video (.mp4)",
                        self.vp_video, browse_file=True)
        self._sep(parent)

        # Steps
        self._lbl(parent, "  PIPELINE STEPS",
                  font=MONO_B, color=C["muted"]).pack(anchor="w", padx=14)
        self._lbl(parent,
                  "  Step 1: install detectron2 wheel from community index",
                  color=C["muted"]).pack(anchor="w", padx=14)
        for i, name in enumerate(D2_STEPS):
            row = tk.Frame(parent, bg=parent.cget("bg"))
            row.pack(fill="x", padx=14, pady=1)
            tk.Checkbutton(row, variable=self.d2_step_checks[i],
                           bg=parent.cget("bg"), fg=C["muted"],
                           selectcolor=C["border"],
                           activebackground=parent.cget("bg"),
                           relief="flat", bd=0).pack(side="left")
            dot = tk.Label(row, text="*", bg=parent.cget("bg"),
                           fg=C["idle"], font=MONO_B)
            dot.pack(side="left")
            self._lbl(row, f"  {name}").pack(side="left")
            self.d2_step_labels.append(dot)

        self._sep(parent)
        self._btn(parent, "[>] RUN ALL (Detectron2)", self._d2_run_all,
                  bg=C["accent"], fg="white", pady=8).pack(
                      fill="x", padx=14)
        self._btn(parent, "[x] STOP", self._stop,
                  bg=C["error"], fg="white").pack(
                      fill="x", padx=14, pady=(6, 2))

    def _select_batch_images(self):
        files = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("All", "*.*")])
        if files:
            self.d2_images = list(files)
            self.batch_count_lbl.config(
                text=f"{len(self.d2_images)} image(s) selected.",
                fg=C["success"])

    def _clear_batch_images(self):
        self.d2_images = []
        self.batch_count_lbl.config(
            text="No images selected.", fg=C["muted"])

    # ---------------------------------------------------
    # VIDEOPOSE3D TAB
    # ---------------------------------------------------
    def _build_vp_tab(self, parent):
        self._lbl(parent, "  INPUT / OUTPUT",
                  font=MONO_B, color=C["muted"]).pack(
                      anchor="w", padx=14, pady=(6, 0))
        self._entry_row(parent, "Input video (.mp4)",
                        self.vp_video,  browse_file=True)
        self._entry_row(parent, "Output file", self.vp_output)
        self._sep(parent)

        self._lbl(parent, "  PIPELINE STEPS",
                  font=MONO_B, color=C["muted"]).pack(anchor="w", padx=14)
        self._lbl(parent,
                  "  Step 1: patch source files + install requirements.txt",
                  color=C["muted"]).pack(anchor="w", padx=14)
        for i, name in enumerate(VP_STEPS):
            row = tk.Frame(parent, bg=parent.cget("bg"))
            row.pack(fill="x", padx=14, pady=1)
            tk.Checkbutton(row, variable=self.vp_step_checks[i],
                           bg=parent.cget("bg"), fg=C["muted"],
                           selectcolor=C["border"],
                           activebackground=parent.cget("bg"),
                           relief="flat", bd=0).pack(side="left")
            dot = tk.Label(row, text="*", bg=parent.cget("bg"),
                           fg=C["idle"], font=MONO_B)
            dot.pack(side="left")
            self._lbl(row, f"  {name}").pack(side="left")
            self.vp_step_labels.append(dot)

        self._sep(parent)
        self._btn(parent, "[>] RUN ALL (VideoPose3D)", self._vp_run_all,
                  bg=C["accent2"], fg="#000", pady=8).pack(
                      fill="x", padx=14)
        self._btn(parent, "[x] STOP", self._stop,
                  bg=C["error"], fg="white").pack(
                      fill="x", padx=14, pady=(6, 2))

    # ---------------------------------------------------
    # LOG
    # ---------------------------------------------------
    def _build_log(self, parent):
        hdr = tk.Frame(parent, bg=C["panel"])
        hdr.pack(fill="x")
        self._lbl(hdr, "  TERMINAL OUTPUT",
                  font=MONO_L, color=C["accent2"]).pack(
                      side="left", padx=14, pady=6)
        self._btn(hdr, "Clear", self._clear_log,
                  bg=C["border"]).pack(side="right", padx=14, pady=4)

        self.log = scrolledtext.ScrolledText(
            parent, bg="#080810", fg=C["text"], font=MONO,
            relief="flat", insertbackground=C["text"], wrap="word")
        self.log.pack(fill="both", expand=True)

        for tag, fg_color, bg_color in [
            ("info",    C["accent2"], None),
            ("success", C["success"], None),
            ("error",   C["error"],   None),
            ("warn",    C["warn"],    None),
            ("cmd",     C["accent"],  None),
            ("step",    "#fff",       C["accent"]),
            ("step2",   "#000",       C["accent2"]),
        ]:
            kw = {"foreground": fg_color}
            if bg_color:
                kw["background"] = bg_color
                kw["font"] = MONO_B
            self.log.tag_config(tag, **kw)

    def _log(self, text, tag=""):
        """Thread-safe: schedule log update on main thread."""
        self.after(0, self._log_ui, text, tag)

    def _log_ui(self, text, tag=""):
        """Must be called on main thread only."""
        self.log.insert("end", text + "\n", tag)
        # Trim old lines to prevent memory buildup
        total = int(self.log.index("end-1c").split(".")[0])
        if total > LOG_MAX_LINES:
            self.log.delete("1.0", f"{total - LOG_MAX_LINES}.0")
        self.log.see("end")

    def _clear_log(self):
        self.log.delete("1.0", "end")

    # ---------------------------------------------------
    # Step indicator  (thread-safe via after())
    # ---------------------------------------------------
    def _set_d2_step(self, idx, state):
        self.after(0, self._set_d2_step_ui, idx, state)

    def _set_d2_step_ui(self, idx, state):
        if idx < len(self.d2_step_labels):
            self.d2_step_labels[idx].config(
                fg={"idle": C["idle"], "running": C["running"],
                    "done": C["done"], "error": C["error"]}.get(
                    state, C["muted"]))

    def _set_vp_step(self, idx, state):
        self.after(0, self._set_vp_step_ui, idx, state)

    def _set_vp_step_ui(self, idx, state):
        if idx < len(self.vp_step_labels):
            self.vp_step_labels[idx].config(
                fg={"idle": C["idle"], "running": C["running"],
                    "done": C["done"], "error": C["error"]}.get(
                    state, C["muted"]))

    # ---------------------------------------------------
    # Command runner  (runs in background thread)
    # ---------------------------------------------------
    def _run_cmd(self, cmd, cwd=None, step_fn=None, state_idx=None):
        """Run a shell command, stream output to log. Returns True/False."""
        self._log(f"\n$ {' '.join(str(c) for c in cmd)}", "cmd")
        if step_fn and state_idx is not None:
            step_fn(state_idx, "running")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                bufsize=0,
                env={**os.environ,
                     "PYTHONIOENCODING": "utf-8",
                     "PYTHONUTF8": "1"},
            )
            self._proc = proc
            for raw in iter(proc.stdout.readline, b""):
                try:
                    line = raw.decode("utf-8", errors="replace")
                except Exception:
                    line = repr(raw)
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

    # ---------------------------------------------------
    # Path helpers
    # ---------------------------------------------------
    @staticmethod
    def _norm(p):
        return os.path.normpath(p) if p else p

    def _py(self):
        return self._norm(self.py_path.get()) or sys.executable

    def _d2(self):
        return self._norm(self.d2_path.get())

    def _vp(self):
        return self._norm(self.vp_path.get())

    def _venv_python(self):
        return str(Path(self._norm(self.venv_path.get())) /
                   "Scripts" / "python.exe")

    def _device(self):
        """Return 'cpu' or 'cuda' based on CUDA selection."""
        return "cpu" if self.torch_cuda_ver.get() == "cpu" else "cuda"

    # ===================================================
    # SHARED VENV STEPS
    # ===================================================
    def _sync_d2_pkg_from_cuda(self):
        cuda = self.torch_cuda_ver.get()
        self.d2_wheel_pkg.set(
            f"detectron2==0.6+fd27788pt2.10.0{cuda}")

    def _auto_detect_d2_pkg(self):
        """Detect installed torch version and fill d2_wheel_pkg."""
        def worker():
            py = self._py()
            if not os.path.isfile(py):
                self._log("[ERROR] Python not found. Create venv first.",
                          "error")
                self._finish_run(); return
            self._log("\nAuto-detecting torch/CUDA version...", "info")
            result = subprocess.run(
                [py, "-c",
                 "import torch; "
                 "v=torch.__version__.split('+')[0]; "
                 "c=torch.version.cuda or 'cpu'; "
                 "cs=c.replace('.','') if c!='cpu' else 'cpu'; "
                 "print('TORCH:'+v); print('CUDA:'+cs)"],
                capture_output=True, text=True,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            torch_ver = cuda_s = None
            for line in (result.stdout + result.stderr).splitlines():
                if line.startswith("TORCH:"):
                    torch_ver = line[6:].strip()
                elif line.startswith("CUDA:"):
                    cuda_s = line[5:].strip()
            if not torch_ver:
                self._log(
                    "[ERROR] Could not detect torch. "
                    "Run 'Install Packages' first.", "error")
                self._finish_run(); return
            if cuda_s == "cpu":
                pkg = f"detectron2==0.6+fd27788pt{torch_ver}cpu"
            else:
                pkg = f"detectron2==0.6+fd27788pt{torch_ver}cu{cuda_s}"
            self.after(0, self.d2_wheel_pkg.set, pkg)
            self._log(f"[OK] Auto-detected: {pkg}", "success")
            self._log(
                "  Browse: https://miropsota.github.io/"
                "torch_packages_builder/detectron2", "info")
            self._finish_run()
        self._start_run(worker)

    def _create_venv(self):
        def worker():
            venv = self.venv_path.get()
            self._log(f"\n=== Create Shared Venv: {venv} ===", "step")
            if os.path.isdir(venv):
                self._log("Venv already exists, skipping creation.", "warn")
            else:
                ok = self._run_cmd([sys.executable, "-m", "venv", venv])
                if not ok:
                    self._log("[ERROR] Failed to create venv.", "error")
                    self._finish_run(); return
                self._log("[OK] Venv created.", "success")
            self.after(0, self.py_path.set, self._venv_python())
            self._log(f"[OK] Python set to: {self._venv_python()}", "info")
            self._finish_run()
        self._start_run(worker)

    def _install_packages(self):
        """Install torch (CUDA) + all deps into the shared venv."""
        def worker():
            self._log("\n=== Install Packages into Shared Venv ===", "step")
            py = self._py()
            if not os.path.isfile(py):
                self._log(f"[ERROR] Python not found: {py}", "error")
                self._log("  Run 'Create Venv' first.", "warn")
                self._finish_run(); return

            cuda = self.torch_cuda_ver.get()
            torch_index = f"https://download.pytorch.org/whl/{cuda}"
            # miropsota detectron2 wheels exist for pt2.10.0
            torch_ver = "2.10.0"

            self._log("Upgrading pip...", "info")
            ok = self._run_cmd(
                [py, "-m", "pip", "install", "--upgrade", "pip"])
            if not ok:
                self._log("[WARN] pip upgrade failed, continuing...", "warn")

            # ── torch (CUDA build from official PyTorch index) ──────────
            self._log(
                f"Installing torch=={torch_ver} ({cuda}) "
                f"from PyTorch index...", "info")
            self._log(f"  Index: {torch_index}", "info")
            ok = self._run_cmd(
                [py, "-m", "pip", "install",
                 "--index-url", torch_index,
                 f"torch=={torch_ver}", "torchvision"])
            if not ok:
                self._log(
                    "[ERROR] torch install failed. "
                    "Check CUDA version selection.", "error")
                self._finish_run(); return

            # ── other deps from PyPI ─────────────────────────────────────
            other = [
                "opencv-python", "matplotlib", "cython",
                "pycocotools", "fvcore", "iopath",
                "omegaconf", "hydra-core", "numpy<2",
            ]
            self._log("Installing other packages...", "info")
            ok = self._run_cmd([py, "-m", "pip", "install"] + other)
            if ok:
                self._log("[OK] All packages installed.", "success")
                self._sync_d2_pkg_from_cuda()
                self._log(
                    f"[OK] D2 wheel pkg set to: "
                    f"{self.d2_wheel_pkg.get()}", "info")
            else:
                self._log(
                    "[ERROR] Some packages failed. Check log.", "error")
            self._finish_run()
        self._start_run(worker)

    # ===================================================
    # DETECTRON2 STEPS
    # ===================================================
    def _d2_step_install(self):
        self._log("\n=== D2 Step 1: Install Detectron2 ===", "step")
        py  = self._py()
        pkg = self.d2_wheel_pkg.get().strip()
        if not pkg:
            self._log(
                "[ERROR] Package name empty. "
                "Use Auto-detect or fill manually.", "error")
            self._set_d2_step(0, "error"); return False

        index_url = "https://miropsota.github.io/torch_packages_builder"
        self._log(f"Installing: {pkg}", "info")
        self._log(f"  Index: {index_url}", "info")
        ok = self._run_cmd(
            [py, "-m", "pip", "install",
             "--extra-index-url", index_url, pkg],
            step_fn=self._set_d2_step, state_idx=0)
        if ok:
            self._log("[OK] Detectron2 installed", "success")
        else:
            self._log("[ERROR] Wheel install failed.", "error")
            self._log(
                "  Run 'Auto-detect' to get the correct package name.", "warn")
            self._log(
                "  Pattern: detectron2==0.6+fd27788pt<torch>cu<cuda>", "warn")
            self._log(
                "  Browse: https://miropsota.github.io/"
                "torch_packages_builder/detectron2", "warn")
        return ok

    def _d2_step_batch_images(self):
        self._log("\n=== D2 Step 2: Batch Image 2D Keypoints ===", "step")
        d2 = self._d2()
        if not self.d2_images:
            self._log(
                "No images selected. Click 'Select Images...' first.",
                "error")
            self._set_d2_step(1, "error"); return False

        self._fix_demo_import(d2)

        # Resolve model via detectron2 model_zoo
        self._log(
            "Resolving keypoint model via model_zoo "
            "(downloads on first run)...", "info")
        d2_fwd = d2.replace("\\", "/")
        resolve_script = (
            "import sys, os; "
            f"sys.path.insert(0, '{d2_fwd}'); "
            "from detectron2 import model_zoo; "
            "url = model_zoo.get_checkpoint_url("
            "'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'); "
            "from detectron2.utils.file_io import PathManager; "
            "local = PathManager.get_local_path(url); "
            "print('MODEL_PATH:' + local)"
        )
        try:
            result = subprocess.run(
                [self._py(), "-c", resolve_script],
                capture_output=True, text=True, cwd=d2,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            model_local = None
            for line in (result.stdout + result.stderr).splitlines():
                self._log(line, "")
                if line.startswith("MODEL_PATH:"):
                    model_local = line[len("MODEL_PATH:"):].strip()
            if not model_local or not os.path.isfile(model_local):
                self._log(
                    "[ERROR] Could not resolve model path via model_zoo.",
                    "error")
                self._log(
                    "  Hint: Run 'Step 1 Install Detectron2' first.",
                    "warn")
                self._set_d2_step(1, "error"); return False
            self._log(f"[OK] Using model: {model_local}", "success")
        except Exception as e:
            self._log(f"[ERROR] model_zoo resolve failed: {e}", "error")
            self._set_d2_step(1, "error"); return False

        out_dir = os.path.join(d2, "demo", "batch_output")
        os.makedirs(out_dir, exist_ok=True)
        self._log(f"Processing {len(self.d2_images)} image(s)...", "info")

        device = self._device()
        if device == "cpu":
            self._log("  [INFO] Running on CPU. This will be slow.", "warn")

        all_ok = True
        for i, img_path in enumerate(self.d2_images):
            if not self.running:
                self._log("[STOPPED] Batch stopped by user.", "warn")
                break
            self._log(
                f"  [{i+1}/{len(self.d2_images)}] "
                f"{Path(img_path).name}", "info")
            ok = self._run_cmd([
                self._py(), "demo/demo.py",
                "--config-file",
                "configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
                "--input",  img_path,
                "--output", out_dir,
                "--device", device,
            ], cwd=d2)
            if not ok:
                self._log(
                    f"  [WARN] Failed on {Path(img_path).name}", "warn")
                all_ok = False

        self._set_d2_step(1, "done" if all_ok else "error")
        if all_ok:
            self._log(
                f"[OK] Batch done. Results in {out_dir}", "success")
        else:
            self._log("[WARN] Some images failed. Check log.", "warn")
        return all_ok

    def _d2_step_infer_video(self):
        self._log("\n=== D2 Step 3: Video 2D Keypoints ===", "step")
        vp    = self._vp()
        video = self.vp_video.get()
        if not video or not os.path.isfile(video):
            self._log("Please select an input video first!", "error")
            self._set_d2_step(2, "error"); return False

        video_dir = os.path.join(vp, "my_videos")
        os.makedirs(video_dir, exist_ok=True)
        dst = os.path.join(video_dir, Path(video).name)
        if os.path.abspath(video) != os.path.abspath(dst):
            shutil.copy2(video, dst)
            self._log(f"Video copied to {dst}", "info")

        os.makedirs(os.path.join(vp, "npz_output"), exist_ok=True)
        self._fix_infer_numpy(vp)

        device = self._device()
        if device == "cpu":
            self._log("  [INFO] Running on CPU. This will be slow.", "warn")

        ok = self._run_cmd([
            self._py(), "inference/infer_video_d2.py",
            "--cfg",        "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
            "--output-dir", "npz_output",
            "--device",     device,
            "my_videos/",
        ], cwd=vp, step_fn=self._set_d2_step, state_idx=2)
        if ok:
            self._log("[OK] 2D keypoints saved to npz_output/", "success")
        return ok

    def _d2_run_all(self):
        checked = [i for i, v in enumerate(self.d2_step_checks) if v.get()]
        if 1 in checked and not self.d2_images:
            self._log(
                "[ERROR] Step 2 (Batch Images) checked "
                "but no images selected!", "error")
            return
        if 2 in checked and not self.vp_video.get():
            self._log(
                "[ERROR] Step 3 (Video 2D) requires "
                "an input video but none selected!", "error")
            return

        step_fns = [
            self._d2_step_install,
            self._d2_step_batch_images,
            self._d2_step_infer_video,
        ]
        def worker():
            for i, fn in enumerate(step_fns):
                if i not in checked:
                    self._log(
                        f"\n[SKIP] D2 {D2_STEPS[i]} (unchecked)", "warn")
                    continue
                if not self.running:
                    self._log("\n[STOPPED]", "warn"); break
                if not fn():
                    self._log("\n[ERROR] Pipeline stopped.", "error"); break
            else:
                self._log(
                    "\n[DONE] All Detectron2 steps completed!", "success")
            self._finish_run()
        self._start_run(worker)

    # ===================================================
    # VIDEOPOSE3D STEPS
    # ===================================================
    def _vp_step_install(self):
        self._log(
            "\n=== VP Step 1: Install VideoPose3D + Patch Files ===",
            "step2")
        vp = self._vp()
        if not os.path.isdir(vp):
            self._log(
                "VideoPose3D directory not found. "
                "Please git clone first.", "error")
            self._set_vp_step(0, "error"); return False

        self._fix_infer_numpy(vp)
        self._fix_viz_fps(vp)
        d2 = self._d2()
        if os.path.isdir(d2):
            self._fix_demo_import(d2)

        req_file = os.path.join(vp, "requirements.txt")
        if os.path.isfile(req_file):
            self._log("Installing requirements.txt...", "info")
            ok = self._run_cmd(
                [self._py(), "-m", "pip", "install",
                 "-r", "requirements.txt"],
                cwd=vp)
        else:
            self._log(
                "No requirements.txt found, "
                "installing matplotlib...", "warn")
            ok = self._run_cmd(
                [self._py(), "-m", "pip", "install", "matplotlib"])

        self._set_vp_step(0, "done" if ok else "error")
        if ok:
            self._log(
                "[OK] VideoPose3D installed and patched", "success")
        return ok

    def _vp_step_infer(self):
        self._log("\n=== VP Step 2: Extract 2D Keypoints ===", "step2")
        vp    = self._vp()
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

        device = self._device()
        if device == "cpu":
            self._log("  [INFO] Running on CPU. This will be slow.", "warn")

        ok = self._run_cmd([
            self._py(), "inference/infer_video_d2.py",
            "--cfg",        "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
            "--output-dir", "npz_output",
            "--image-ext",  "mp4",
            "--device", device,
            "my_videos/",
        ], cwd=vp, step_fn=self._set_vp_step, state_idx=1)
        if ok:
            self._log("[OK] 2D keypoints saved to npz_output/", "success")
        return ok

    def _vp_step_prepare(self):
        self._log("\n=== VP Step 3: Convert Format ===", "step2")
        vp = self._vp()
        ok = self._run_cmd([
            self._py(), "prepare_data_2d_custom.py",
            "-i", "../npz_output", "-o", "myvideos",
        ], cwd=os.path.join(vp, "data"),
           step_fn=self._set_vp_step, state_idx=2)
        if ok:
            self._log(
                "[OK] data/data_2d_custom_myvideos.npz created", "success")
        return ok

    def _vp_step_download(self):
        self._log(
            "\n=== VP Step 4: Download Pretrained Model ===", "step2")
        vp       = self._vp()
        ckpt_dir = os.path.join(vp, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        model_name = "pretrained_h36m_detectron_coco.bin"
        model      = os.path.join(ckpt_dir, model_name)
        url        = ("https://dl.fbaipublicfiles.com/video-pose-3d/"
                      + model_name)

        if os.path.isfile(model):
            self._log("Model already exists, skipping download.", "warn")
            self._set_vp_step(3, "done"); return True

        ok = self._download_file_python(
            url, model, step_fn=self._set_vp_step, state_idx=3)
        if ok:
            self._log("[OK] Pretrained model downloaded", "success")
        return ok

    def _download_file_python(self, url, dest_path,
                              step_fn=None, state_idx=None):
        """Download via urllib with progress (no wget needed on Windows)."""
        self._log(f"\n$ [urllib download] {url}", "cmd")
        if step_fn and state_idx is not None:
            step_fn(state_idx, "running")
        try:
            last_pct = [-1]
            def reporthook(block_num, block_size, total_size):
                if total_size <= 0:
                    return
                downloaded = min(block_num * block_size, total_size)
                pct = int(downloaded * 100 / total_size)
                if pct != last_pct[0] and pct % 10 == 0:
                    last_pct[0] = pct
                    mb_done  = downloaded / 1024 / 1024
                    mb_total = total_size  / 1024 / 1024
                    self._log(
                        f"  {pct}%  "
                        f"({mb_done:.1f} / {mb_total:.1f} MB)", "info")
            urllib.request.urlretrieve(url, dest_path, reporthook)
            self._log("  100% download complete.", "success")
            if step_fn and state_idx is not None:
                step_fn(state_idx, "done")
            return True
        except Exception as e:
            self._log(f"[ERROR] Download failed: {e}", "error")
            if os.path.isfile(dest_path):
                try: os.remove(dest_path)
                except OSError: pass
            if step_fn and state_idx is not None:
                step_fn(state_idx, "error")
            return False

    def _vp_step_run3d(self):
        self._log("\n=== VP Step 5: 3D Inference & Output ===", "step2")
        vp    = self._vp()
        video = self.vp_video.get()
        if not video:
            self._log("Please select an input video first!", "error")
            self._set_vp_step(4, "error"); return False

        self._fix_viz_fps(vp)

        subject = Path(video).name
        output  = self.vp_output.get() or "output_videos/output_3d.mp4"
        ok = self._run_cmd([
            self._py(), "run.py",
            "-d", "custom", "-k", "myvideos", "-arc", "3,3,3,3,3",
            "-c", "checkpoint",
            "--evaluate", "pretrained_h36m_detectron_coco.bin",
            "--render",
            "--viz-subject",    subject,
            "--viz-action",     "custom",
            "--viz-camera",     "0",
            "--viz-video",      video,
            "--viz-output",     output,
            "--viz-size",       "5",
            "--viz-downsample", "2",
        ], cwd=vp, step_fn=self._set_vp_step, state_idx=4)
        if ok:
            self._log(
                f"[OK] Output saved: {os.path.join(vp, output)}", "success")
        return ok

    def _vp_run_all(self):
        checked = [i for i, v in enumerate(self.vp_step_checks) if v.get()]
        if (1 in checked or 4 in checked) and not self.vp_video.get():
            self._log(
                "[ERROR] VP Step 2/5 requires an input video "
                "but none selected!", "error")
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
                    self._log(
                        f"\n[SKIP] VP {VP_STEPS[i]} (unchecked)", "warn")
                    continue
                if not self.running:
                    self._log("\n[STOPPED]", "warn"); break
                if not fn():
                    self._log("\n[ERROR] Pipeline stopped.", "error"); break
            else:
                self._log(
                    "\n[DONE] All VideoPose3D steps completed!", "success")
            self._finish_run()
        self._start_run(worker)

    # ===================================================
    # PATCH HELPERS
    # ===================================================
    def _fix_demo_import(self, d2):
        demo_py = os.path.join(d2, "demo", "demo.py")
        if not os.path.isfile(demo_py): return
        with open(demo_py, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
        if "vision.fair" in src:
            src = src.replace(
                "from vision.fair.detectron2.demo.predictor"
                " import VisualizationDemo",
                "from predictor import VisualizationDemo")
            with open(demo_py, "w", encoding="utf-8") as f:
                f.write(src)
            self._log(
                "[FIXED] demo.py: corrected import path", "warn")

    def _fix_infer_numpy(self, vp):
        infer_py = os.path.join(vp, "inference", "infer_video_d2.py")
        if not os.path.isfile(infer_py): return
        with open(infer_py, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
        if "dtype=object" not in src:
            src = src.replace(
                "np.savez_compressed(out_name, boxes=boxes, "
                "segments=segments, keypoints=keypoints, metadata=metadata)",
                "np.savez_compressed(out_name, "
                "boxes=np.array(boxes, dtype=object), "
                "segments=np.array(segments, dtype=object), "
                "keypoints=np.array(keypoints, dtype=object), "
                "metadata=metadata)")
            with open(infer_py, "w", encoding="utf-8") as f:
                f.write(src)
            self._log(
                "[FIXED] infer_video_d2.py: numpy dtype fix", "warn")

    def _fix_viz_fps(self, vp):
        viz_py = os.path.join(vp, "common", "visualization.py")
        if not os.path.isfile(viz_py): return
        with open(viz_py, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
        if "fps = fps or 30" not in src:
            src = src.replace(
                "fps /= downsample",
                "fps = fps or 30\n    fps /= downsample")
            with open(viz_py, "w", encoding="utf-8") as f:
                f.write(src)
            self._log(
                "[FIXED] visualization.py: fps=None fallback", "warn")

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