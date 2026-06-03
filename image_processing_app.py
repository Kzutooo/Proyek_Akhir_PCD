#!/usr/bin/env python3
"""
Aplikasi Pengolahan Citra Digital
Tugas Akhir - Pengolahan Citra Digital
Fitur: Input/Tampil Gambar, Grayscale, Biner, Aritmatika, Logika,
    Histogram, Konvolusi (Filter), Morfologi
"""

import tkinter as tk
from typing import Optional
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
from PIL import Image, ImageTk
import os

# Cek dependensi opsional
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ─────────────────────────────────────────────
# FUNGSI PEMBANTU
# ─────────────────────────────────────────────

def pil_to_array(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.float64)

def array_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode='L')
    return Image.fromarray(arr, mode='RGB')

def resize_for_display(img: Image.Image, max_w=380, max_h=380) -> Image.Image:
    w, h = img.size
    ratio = min(max_w / w, max_h / h, 1.0)
    return img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

def to_gray_array(img: Image.Image) -> np.ndarray:
    """Kembalikan array 2D grayscale float64."""
    if img.mode == 'L':
        return pil_to_array(img)
    rgb = np.array(img.convert('RGB'), dtype=np.float64)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


# ─────────────────────────────────────────────
# KONVOLUSI MANUAL
# ─────────────────────────────────────────────

def convolve2d_manual(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Konvolusi 2D tanpa scipy (padding='reflect', valid for 2-D array)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='reflect')
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return out

def apply_kernel(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if SCIPY_AVAILABLE:
        return ndimage.convolve(gray, kernel, mode='reflect')
    return convolve2d_manual(gray, kernel)


# ─────────────────────────────────────────────
# MORFOLOGI MANUAL
# ─────────────────────────────────────────────

def morph_dilasi(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    kh, kw = se.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(binary, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
    out = np.zeros_like(binary)
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            if np.any(region[se == 1] == 1):
                out[i, j] = 1
    return out

def morph_erosi(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    kh, kw = se.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(binary, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
    out = np.zeros_like(binary)
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            if np.all(region[se == 1] == 1):
                out[i, j] = 1
    return out


# ─────────────────────────────────────────────
# ELEMEN PENSTRUKTUR (SE)
# ─────────────────────────────────────────────

SE_LIBRARY = {
    "Persegi 3×3 (Square)": np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.uint8),
    
    "Cross / Plus 3×3": np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)
}

# ─────────────────────────────────────────────
# APLIKASI UTAMA
# ─────────────────────────────────────────────

class ImageProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplikasi Pengolahan Citra Digital - Tugas Akhir")
        self.configure(bg="#0f1117")
        self.resizable(True, True)
        self.geometry("1200x820")

        self.original_image = None  # type: Optional[Image.Image]
        self.result_image   = None  # type: Optional[Image.Image]

        self._build_styles()
        self._build_ui()

    # ── Styles ──
    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        bg = "#0f1117"
        panel = "#1a1d27"
        accent = "#00d4aa"
        fg = "#e8eaf0"
        style.configure('TFrame', background=bg)
        style.configure('Panel.TFrame', background=panel)
        style.configure('TLabel', background=bg, foreground=fg,
                        font=('Times New Roman', 10))
        style.configure('Header.TLabel', background=bg, foreground=accent,
                        font=('Times New Roman', 11, 'bold'))
        style.configure('TButton', background=panel, foreground=fg,
                        font=('Times New Roman', 9), relief='flat', padding=6)
        style.map('TButton',
                background=[('active', accent)],
                foreground=[('active', '#0f1117')])
        style.configure('Accent.TButton', background=accent, foreground='#0f1117',
                        font=('Times New Roman', 9, 'bold'), padding=6)
        style.map('Accent.TButton',
                background=[('active', '#00ffcc')],
                foreground=[('active', '#0f1117')])
        style.configure('TCombobox', fieldbackground=panel, background=panel,
                        foreground=fg, selectbackground=accent)
        style.configure('TNotebook', background=bg, tabmargins=[2, 5, 2, 0])
        style.configure('TNotebook.Tab', background=panel, foreground=fg,
                        padding=[12, 5], font=('Times New Roman', 9))
        style.map('TNotebook.Tab',
                background=[('selected', accent)],
                foreground=[('selected', '#0f1117')])
        style.configure('TScale', background=bg, troughcolor=panel)

    # ── Main UI ──
    def _build_ui(self):
        # Title bar
        title_bar = ttk.Frame(self)
        title_bar.pack(fill='x', padx=0, pady=0)
        tk.Label(title_bar, text="  DIGITAL IMAGE PROCESSING KELOMPOK HEKSAGRAM",
                bg="#0f1117", fg="#00d4aa",
                font=('Times New Roman', 14, 'bold')).pack(pady=10)

        main = ttk.Frame(self)
        main.pack(fill='both', expand=True, padx=16, pady=(0, 16))

        # Left panel – controls (scrollable)
        # Dilebarkan sedikit menjadi 280 agar scrollbar tidak menutupi ujung elemen
        left_outer = ttk.Frame(main, style='Panel.TFrame', width=280)
        left_outer.pack(side='left', fill='y', padx=(0, 12))
        left_outer.pack_propagate(False)

        _cv = tk.Canvas(left_outer, bg="#1a1d27", highlightthickness=0)
        _sb = ttk.Scrollbar(left_outer, orient='vertical', command=_cv.yview)
        left = ttk.Frame(_cv, style='Panel.TFrame')
        
        # Simpan ID window untuk kontrol ukuran dinamis
        frame_id = _cv.create_window((0, 0), window=left, anchor='nw')

        # Update scrollregion saat tinggi konten bertambah
        left.bind("<Configure>", lambda e: _cv.configure(scrollregion=_cv.bbox("all")))
        
        # Paksa frame internal mengikuti lebar canvas agar elemen tidak bertumpuk
        _cv.bind("<Configure>", lambda e: _cv.itemconfig(frame_id, width=e.width))

        _cv.configure(yscrollcommand=_sb.set)
        _sb.pack(side='right', fill='y')
        _cv.pack(side='left', fill='both', expand=True)
        
        # Scroll dengan mouse wheel
        def _on_mousewheel(e):
            _cv.yview_scroll(int(-1*(e.delta/120)), "units")
        _cv.bind_all("<MouseWheel>", _on_mousewheel)
        
        self._build_left_panel(left)

        # Paksa kalkulasi layout sebelum menentukan area scroll akhir
        self.update_idletasks()
        _cv.configure(scrollregion=_cv.bbox("all"))

        # Right – image display + tabs
        right = ttk.Frame(main)
        right.pack(side='left', fill='both', expand=True)
        self._build_right_panel(right)

    # ── Left Panel ──
    def _build_left_panel(self, parent):
        pad = {'padx': 12, 'pady': 4}

        ttk.Label(parent, text="[ INPUT Gambar]", style='Header.TLabel',
                  background="#1a1d27").pack(anchor='w', **pad)

        ttk.Button(parent, text="[+] Buka Gambar",
                style='Accent.TButton',
                   command=self.load_image).pack(fill='x', **pad)
        ttk.Button(parent, text="[X] Hapus Gambar",
                   command=self.clear_main_image).pack(fill='x', **pad)

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=8, padx=12)
        ttk.Label(parent, text="[ PROSES DASAR ]", style='Header.TLabel',
                  background="#1a1d27").pack(anchor='w', **pad)

        ttk.Button(parent, text="[G] Grayscale",
                   command=self.to_grayscale).pack(fill='x', **pad)
        ttk.Button(parent, text="[B] Citra Biner",
                   command=self.to_binary).pack(fill='x', **pad)

        # Threshold slider
        tf = ttk.Frame(parent, style='Panel.TFrame')
        tf.pack(fill='x', **pad)
        ttk.Label(tf, text="Threshold:", background="#1a1d27",
                foreground="#aab0c0").pack(side='left')
        self.thresh_var = tk.IntVar(value=128)
        ttk.Scale(tf, from_=0, to=255, variable=self.thresh_var,
                orient='horizontal').pack(side='left', fill='x', expand=True)
        ttk.Label(tf, textvariable=self.thresh_var, background="#1a1d27",
                foreground="#00d4aa", width=4).pack(side='left')

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=8, padx=12)
        ttk.Label(parent, text="[ ARITMATIKA ]", style='Header.TLabel',
                  background="#1a1d27").pack(anchor='w', **pad)

        ops_frame = ttk.Frame(parent, style='Panel.TFrame')
        ops_frame.pack(fill='x', **pad)
        
        # Baris 1: Tambah dan Kurang
        row1_frame = ttk.Frame(ops_frame, style='Panel.TFrame')
        row1_frame.pack(fill='x', pady=2)
        
        ttk.Button(row1_frame, text="Tambah",
                command=lambda: self.arith_op("Tambah")).pack(
                    side='left', expand=True, fill='x', padx=2)
        ttk.Button(row1_frame, text="Kurang",
                command=lambda: self.arith_op("Kurang")).pack(
                    side='left', expand=True, fill='x', padx=2)

        # Baris 2: Kali dan Bagi
        row2_frame = ttk.Frame(ops_frame, style='Panel.TFrame')
        row2_frame.pack(fill='x', pady=2)
        
        ttk.Button(row2_frame, text="Kali",
                command=lambda: self.arith_op("Kali")).pack(
                    side='left', expand=True, fill='x', padx=2)
        ttk.Button(row2_frame, text="Bagi",
                command=lambda: self.arith_op("Bagi")).pack(
                    side='left', expand=True, fill='x', padx=2)

        ttk.Label(parent, text="  Nilai skalar :",
                background="#1a1d27", foreground="#aab0c0",
                font=('Times New Roman', 8)).pack(anchor='w', padx=12)
        self.scalar_var = tk.IntVar(value=50)
        sf = ttk.Frame(parent, style='Panel.TFrame')
        sf.pack(fill='x', **pad)
        ttk.Scale(sf, from_=1, to=255, variable=self.scalar_var,
                orient='horizontal').pack(side='left', fill='x', expand=True)
        ttk.Label(sf, textvariable=self.scalar_var, background="#1a1d27",
                foreground="#00d4aa", width=4).pack(side='left')

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=8, padx=12)
        ttk.Label(parent, text="[ LOGIKA ]", style='Header.TLabel',
                  background="#1a1d27").pack(anchor='w', **pad)
        logic_frame = ttk.Frame(parent, style='Panel.TFrame')
        logic_frame.pack(fill='x', **pad)
        for op in ["AND", "OR", "NOT", "XOR"]:
            ttk.Button(logic_frame, text=op,
                    command=lambda o=op: self.logic_op(o)).pack(
                        side='left', expand=True, fill='x', padx=2)

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=8, padx=12)
        ttk.Label(parent, text="[ HISTOGRAM ]", style='Header.TLabel',
                  background="#1a1d27").pack(anchor='w', **pad)
        ttk.Button(parent, text="[H] Tampilkan Histogram",
                   command=self.show_histogram).pack(fill='x', **pad)

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=8, padx=12)
        ttk.Label(parent, text="[ KONVOLUSI / FILTER ]", style='Header.TLabel',
                  background="#1a1d27").pack(anchor='w', **pad)
        self.filter_var = tk.StringVar(value="Sharpening")
        filters = ["Sharpening", "Blurring", "Edge Detection"]
        ttk.Combobox(parent, textvariable=self.filter_var,
                     values=filters, state='readonly').pack(fill='x', **pad)
        ttk.Button(parent, text="[>] Terapkan Filter",
                   command=self.apply_filter).pack(fill='x', **pad)

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=8, padx=12)
        ttk.Label(parent, text="[ MORFOLOGI ]", style='Header.TLabel',
                  background="#1a1d27").pack(anchor='w', **pad)
        self.morph_op_var = tk.StringVar(value="Dilasi")
        ttk.Combobox(parent, textvariable=self.morph_op_var,
                     values=["Dilasi", "Erosi"], state='readonly').pack(fill='x', **pad)
        self.se_var = tk.StringVar(value=list(SE_LIBRARY.keys())[0])
        ttk.Combobox(parent, textvariable=self.se_var,
                     values=list(SE_LIBRARY.keys()), state='readonly').pack(fill='x', **pad)
        ttk.Button(parent, text="[>] Terapkan Morfologi",
                   command=self.apply_morphology).pack(fill='x', **pad)

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=8, padx=12)
        ttk.Button(parent, text="[S] Simpan Hasil",
                style='Accent.TButton',
                   command=self.save_result).pack(fill='x', **pad)
        
        # Beri padding bawah ekstra (24px) pada elemen terakhir 
        # agar tidak terpotong oleh batas bawah area scroll.
        ttk.Button(parent, text="[R] Reset",
                   command=self.reset).pack(fill='x', padx=12, pady=(4, 24))

    # ── Right Panel ──
    def _build_right_panel(self, parent):
        # Image display area
        img_row = ttk.Frame(parent)
        img_row.pack(fill='x')

        # Original
        orig_frame = ttk.Frame(img_row, style='Panel.TFrame')
        orig_frame.pack(side='left', fill='both', expand=True, padx=(0, 6))
        ttk.Label(orig_frame, text="GAMBAR ASLI",
                background="#1a1d27", foreground="#00d4aa",
                font=('Times New Roman', 9, 'bold')).pack(pady=(8, 4))
        self.orig_canvas = tk.Canvas(orig_frame, bg="#0d0f18",
                                    highlightthickness=0, width=340, height=340)
        self.orig_canvas.pack(padx=8, pady=8)
        self.orig_info = ttk.Label(orig_frame, text="—",
                                background="#1a1d27", foreground="#aab0c0",
                                font=('Times New Roman', 8))
        self.orig_info.pack(pady=(0, 8))

        # Result
        res_frame = ttk.Frame(img_row, style='Panel.TFrame')
        res_frame.pack(side='left', fill='both', expand=True)
        ttk.Label(res_frame, text="HASIL PROSES",
                background="#1a1d27", foreground="#00d4aa",
                font=('Times New Roman', 9, 'bold')).pack(pady=(8, 4))
        self.res_canvas = tk.Canvas(res_frame, bg="#0d0f18",
                                    highlightthickness=0, width=340, height=340)
        self.res_canvas.pack(padx=8, pady=8)
        self.res_info = ttk.Label(res_frame, text="—",
                                background="#1a1d27", foreground="#aab0c0",
                                font=('Times New Roman', 8))
        self.res_info.pack(pady=(0, 8))

        # Log / status bar
        log_frame = ttk.Frame(parent, style='Panel.TFrame')
        log_frame.pack(fill='x', pady=(10, 0))
        ttk.Label(log_frame, text="LOG:", background="#1a1d27",
                foreground="#00d4aa", font=('Times New Roman', 8, 'bold')).pack(
                    side='left', padx=8)
        self.log_var = tk.StringVar(value="Siap. Silakan buka gambar.")
        ttk.Label(log_frame, textvariable=self.log_var,
                background="#1a1d27", foreground="#e8eaf0",
                font=('Times New Roman', 8)).pack(side='left', padx=4)

    # ── Display helpers ──
    def _show_on_canvas(self, canvas, img: Image.Image) -> None:
        disp = resize_for_display(img, 340, 340)
        tk_img = ImageTk.PhotoImage(disp)
        canvas._tk_img = tk_img
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 10:
            cw, ch = 340, 340
        canvas.delete('all')
        canvas.create_image(cw // 2, ch // 2, anchor='center', image=tk_img)

    def _log(self, msg: str):
        self.log_var.set(msg)
        self.update_idletasks()

    def _img_info(self, img: Image.Image) -> str:
        return f"{img.width}×{img.height} | Mode: {img.mode}"

    # ── Image I/O ──
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Gambar", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("Semua", "*.*")])
        if not path:
            return
        self.original_image = Image.open(path).convert('RGB')
        self._show_on_canvas(self.orig_canvas, self.original_image)
        self.orig_info.config(text=self._img_info(self.original_image))
        self._log(f"Gambar dimuat: {os.path.basename(path)}")

    def clear_main_image(self):
        self.original_image = None
        self.orig_canvas.delete('all')
        self.orig_info.config(text="—")
        # Bersihkan juga hasil karena bergantung pada gambar utama
        self.result_image = None
        self.res_canvas.delete('all')
        self.res_info.config(text="—")
        self._log("Gambar utama dihapus.")

    def save_result(self):
        if self.result_image is None:
            messagebox.showwarning("Peringatan", "Belum ada hasil untuk disimpan.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        if path:
            self.result_image.save(path)
            self._log(f"Hasil disimpan: {os.path.basename(path)}")

    def reset(self):
        self.result_image = None
        self.res_canvas.delete('all')
        self.res_info.config(text="—")
        self._log("Reset dilakukan.")

    def _require_image(self) -> bool:
        if self.original_image is None:
            messagebox.showwarning("Peringatan", "Silakan buka gambar terlebih dahulu.")
            return False
        return True

    def _show_result(self, img: Image.Image, label: str):
        self.result_image = img
        self._show_on_canvas(self.res_canvas, img)
        self.res_info.config(text=self._img_info(img))
        self._log(label)

    # ── Proses Dasar ──
    def to_grayscale(self):
        if not self._require_image():
            return
        gray = self.original_image.convert('L')
        self._show_result(gray, "Konversi ke Grayscale selesai.")

    def to_binary(self):
        if not self._require_image():
            return
        t = self.thresh_var.get()
        gray = to_gray_array(self.original_image)
        binary = (gray >= t).astype(np.uint8) * 255
        result = Image.fromarray(binary, mode='L')
        self._show_result(result, f"Citra Biner (threshold={t}) selesai.")

    # ── Operasi Aritmatika ──
    def arith_op(self, op: str):
        if not self._require_image():
            return
        a = pil_to_array(self.original_image)
        b = float(self.scalar_var.get())

        if op == "Tambah":
            res = a + b
        elif op == "Kurang":
            res = a - b
        elif op == "Kali":
            res = a * (b / 50.0)
        elif op == "Bagi":
            res = a / (b if b != 0 else 1) * 128
        else:
            return

        self._show_result(array_to_pil(res), f"Operasi Aritmatika [{op}] selesai.")

    # ── Operasi Logika ──
    def logic_op(self, op: str):
        if not self._require_image():
            return
        a = pil_to_array(self.original_image).astype(np.uint8)
        if op == "NOT":
            res = ~a
            self._show_result(array_to_pil(res.astype(np.float64)),
                            "Operasi Logika [NOT] selesai.")
            return

        # AND / OR / XOR: operasikan gambar dengan versi threshold-nya sendiri
        t = self.thresh_var.get()
        gray = to_gray_array(self.original_image)
        mask = ((gray >= t).astype(np.uint8) * 255)
        if a.ndim == 3:
            mask = np.stack([mask, mask, mask], axis=2).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        if op == "AND":
            res = a & mask
        elif op == "OR":
            res = a | mask
        elif op == "XOR":
            res = a ^ mask
        else:
            return
        self._show_result(array_to_pil(res.astype(np.float64)),
                        f"Operasi Logika [{op}] selesai.")

    # ── Histogram ──
    def show_histogram(self):
        if not self._require_image():
            return
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showerror("Error",
                "matplotlib tidak tersedia.\nInstall: pip install matplotlib")
            return

        img_arr = np.array(self.original_image.convert('RGB'))
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        fig.patch.set_facecolor('#0f1117')
        colors_map = [('#e74c3c', 'Red'), ('#2ecc71', 'Green'),
                    ('#3498db', 'Blue'), ('#00d4aa', 'Grayscale')]
        for i, (color, label) in enumerate(colors_map):
            ax = axes[i]
            ax.set_facecolor('#1a1d27')
            ax.tick_params(colors='#aab0c0')
            ax.spines[:].set_color('#2a2d3a')
            ax.title.set_color('#00d4aa')
            ax.set_title(label)
            if i < 3:
                channel = img_arr[:, :, i]
            else:
                channel = np.array(self.original_image.convert('L'))
            ax.hist(channel.ravel(), bins=256, range=(0, 256),
                    color=color, alpha=0.85, edgecolor='none')
        plt.tight_layout()

        win = tk.Toplevel(self)
        win.title("Histogram")
        win.configure(bg='#0f1117')
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self._log("Histogram ditampilkan.")

    # ── Filter / Konvolusi ──
    def apply_filter(self):
        if not self._require_image():
            return
        name = self.filter_var.get()
        KERNELS = {
            "Sharpening": np.array(
                [[0,-1, 0],[-1, 5,-1],[0,-1, 0]], dtype=np.float64),
            "Blurring": np.array(
                [[1, 2, 1],[2, 4, 2],[1, 2, 1]], dtype=np.float64) / 16,
            "Edge Detection": np.array(
                [[-1,-2,-1],[0, 0, 0],[1, 2, 1]], dtype=np.float64),
        }
        kernel = KERNELS[name]
        gray = to_gray_array(self.original_image)
        filtered = apply_kernel(gray, kernel)
        result = array_to_pil(filtered)
        self._show_result(result, f"Filter [{name}] diterapkan.")

    # ── Morfologi ──
    def apply_morphology(self):
        if not self._require_image():
            return
        op      = self.morph_op_var.get()
        se_name = self.se_var.get()
        se      = SE_LIBRARY[se_name]
        t       = self.thresh_var.get()

        self._log("Memproses morfologi, harap tunggu...")
        self.update_idletasks()

        # Hitung biner, dilasi, erosi
        gray    = to_gray_array(self.original_image)
        binary  = (gray >= t).astype(np.uint8)
        bin_img = Image.fromarray((binary * 255).astype(np.uint8), mode='L')
        dil_img = Image.fromarray((morph_dilasi(binary, se) * 255).astype(np.uint8), mode='L')
        ero_img = Image.fromarray((morph_erosi(binary, se)  * 255).astype(np.uint8), mode='L')

        # Tampilkan hasil yang dipilih di panel utama
        if op == "Dilasi":
            self._show_result(dil_img, f"Morfologi [Dilasi] dengan SE [{se_name}] selesai.")
        else:
            self._show_result(ero_img, f"Morfologi [Erosi] dengan SE [{se_name}] selesai.")

        # ── Popup perbandingan 3 panel: Biner | Dilasi | Erosi ──
        win = tk.Toplevel(self)
        win.title(f"Hasil Morfologi  |  SE: {se_name}  |  Threshold: {t}")
        win.configure(bg="#0f1117")
        win.resizable(True, True)
        win.geometry("980x530")

        tk.Label(win,
                text=f"SE: {se_name}   |   Threshold: {t}",
                bg="#0f1117", fg="#00d4aa",
                font=('Times New Roman', 11, 'bold')).pack(pady=(10, 4))

        panels_frame = ttk.Frame(win)
        panels_frame.pack(fill='both', expand=True, padx=12, pady=4)

        PANEL_W, PANEL_H = 290, 290
        panel_configs = [
            ("BINER (ASLI)", bin_img, "#aab0c0"),
            ("DILASI",       dil_img, "#00d4aa"),
            ("EROSI",        ero_img, "#e74c3c"),
        ]
        win._tk_imgs = []
        for title, img, color in panel_configs:
            frame = tk.Frame(panels_frame, bg="#1a1d27",
                            highlightbackground=color, highlightthickness=2)
            frame.pack(side='left', fill='both', expand=True, padx=6)
            tk.Label(frame, text=title, bg="#1a1d27", fg=color,
                    font=('Times New Roman', 10, 'bold')).pack(pady=(8, 4))
            c = tk.Canvas(frame, bg="#0d0f18",
                        highlightthickness=0, width=PANEL_W, height=PANEL_H)
            c.pack(padx=6, pady=4)
            disp   = resize_for_display(img, PANEL_W, PANEL_H)
            tk_img = ImageTk.PhotoImage(disp)
            win._tk_imgs.append(tk_img)
            c.create_image(PANEL_W // 2, PANEL_H // 2, anchor='center', image=tk_img)
            px = int(np.sum(np.array(img) > 0))
            tk.Label(frame, text=f"Piksel aktif: {px:,}",
                    bg="#1a1d27", fg="#aab0c0",
                    font=('Times New Roman', 8)).pack(pady=(0, 6))

        # Tombol bawah popup
        btn_row = ttk.Frame(win)
        btn_row.pack(fill='x', padx=12, pady=(6, 12))

        def _save(img, suffix):
            path = filedialog.asksaveasfilename(
                parent=win, initialfile=f"morfologi_{suffix}.png",
                defaultextension=".png",
                filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("BMP","*.bmp")])
            if path:
                img.save(path)
                self._log(f"Disimpan: {os.path.basename(path)}")

        ttk.Button(btn_row, text="Simpan Biner",
                command=lambda: _save(bin_img, "biner")).pack(side='left', padx=4)
        ttk.Button(btn_row, text="Simpan Dilasi",
                command=lambda: _save(dil_img, "dilasi")).pack(side='left', padx=4)
        ttk.Button(btn_row, text="Simpan Erosi",
                command=lambda: _save(ero_img, "erosi")).pack(side='left', padx=4)
        ttk.Button(btn_row, text="Pakai Dilasi", style='Accent.TButton',
                command=lambda: (
                    self._show_result(dil_img, f"Dilasi SE [{se_name}] selesai."),
                    win.destroy())).pack(side='right', padx=4)
        ttk.Button(btn_row, text="Pakai Erosi",
                command=lambda: (
                    self._show_result(ero_img, f"Erosi SE [{se_name}] selesai."),
                    win.destroy())).pack(side='right', padx=4)
        ttk.Button(btn_row, text="Tutup",
                command=win.destroy).pack(side='right', padx=4)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = ImageProcessingApp()
    app.mainloop()