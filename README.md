# AugmentationPipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.6%2B-green)](https://www.python.org/)

## 📖 Deskripsi

`AugmentationPipeline` adalah sebuah toolkit Python profesional untuk menghasilkan augmentasi data sinyal (misalnya kromatografi) dengan beragam transformasi probabilistik dan non-linear. Dirancang agar mudah di-integrasikan dalam workflow analisis data dan machine learning, baik untuk eksperimen laboratorium maupun produksi.

**Fitur utama**:

* Penambahan noise Gaussian
* Pergeseran (shift) waktu retensi
* Scaling intensitas sinyal
* Drift baseline linier & non-linear
* Spectral warp linier & non-linear
* Smoothing Savitzky–Golay
* Reproduksibilitas via seed

---

## ⚙️ Instalasi

Jalankan perintah berikut di terminal:

```bash
pip install numpy pandas scipy openpyxl
```

> **Catatan**:
>
> * `openpyxl` digunakan untuk membaca/menulis file `.xlsx`.
> * Jika Anda menggunakan virtual environment, aktifkan terlebih dahulu (`python3 -m venv venv && source venv/bin/activate`).

---

## 📂 Struktur Proyek

```
AugmentationPipeline/       ← Root repository (GitHub: @Arifmaulanaazis/AugmentationPipeline)
├── augmentation.py         ← Script utama pipeline
├── example.xlsx            ← Contoh dataset input (Time & Intensity)
├── output.xlsx             ← Contoh dataset output augmentasi (Time & Intensity)
├── README.md               ← Dokumentasi (ini)
└── LICENSE                 ← Lisensi MIT
```

---

## 🚀 Cara Penggunaan

### 1. Menggunakan sebagai CLI

Jalankan `augmentation.py` dari command line:

```bash
python augmentation.py \
  -i example.xlsx \
  -x Time \
  -y Intensity \
  -n 5 \
  -o output.xlsx \
  -s 42 \
  --baseline_nonlinear \
  --warp_nonlinear
```

#### Penjelasan Argumen

| Opsi                           | Singkatan | Tipe      | Default         | Deskripsi                                        |
| ------------------------------ | --------- | --------- | --------------- | ------------------------------------------------ |
| `-i`, `--input`                |           | `str`     | **(wajib)**     | Path file input (`.csv` atau `.xlsx`).           |
| `-x`, `--input_x`              |           | `str`     | **(wajib)**     | Nama kolom sumbu X.                              |
| `-y`, `--input_y`              |           | `str`     | **(wajib)**     | Nama kolom Y (pisahkan koma jika >1).            |
| `-o`, `--output`               |           | `str`     | `output.xlsx`   | Path file output (`.xlsx`, `.csv`, atau `.npz`). |
| `-n`, `--n_samples`            |           | `int`     | `1`             | Jumlah sampel augmentasi.                        |
| `-s`, `--seed`                 |           | `int`     | `None`          | Seed untuk reproduksibilitas.                    |
| `-ka`, `--kolom_augmentasi`    |           | `str`     | `Augmentasi`    | Prefix kolom hasil augmentasi.                   |
| `-tr`, `--X_shift_range`       |           | `float` 2 | `[-0.1, 0.1]`   | Rentang pergeseran X.                            |
| `-tp`, `--X_shift_p`           |           | `float`   | `0.5`           | Probabilitas apply X shift.                      |
| `-nr`, `--noise_std_range`     |           | `float` 2 | `[0.005, 0.02]` | Rentang σ noise Gaussian.                        |
| `-np`, `--noise_std_p`         |           | `float`   | `0.5`           | Probabilitas menambahkan noise.                  |
| `-scr`, `--scale_range`        |           | `float` 2 | `[0.95, 1.05]`  | Rentang faktor scaling.                          |
| `-scp`, `--scale_p`            |           | `float`   | `0.5`           | Probabilitas apply scaling.                      |
| `-br`, `--baseline_range`      |           | `float` 2 | `[-0.01, 0.01]` | Rentang total drift baseline linier.             |
| `-bp`, `--baseline_p`          |           | `float`   | `0.5`           | Probabilitas apply baseline drift.               |
| `-bln`, `--baseline_nonlinear` | action    |           | `False`         | Gunakan drift baseline non-linear.               |
| `-wr`, `--warp_range`          |           | `float` 2 | `[-0.03, 0.03]` | Rentang spectral warp linier.                    |
| `-wp`, `--warp_p`              |           | `float`   | `0.5`           | Probabilitas apply warp.                         |
| `-wnl`, `--warp_nonlinear`     | action    |           | `False`         | Gunakan spectral warp non-linear.                |
| `-sw`, `--smooth_window`       |           | `int`     | `7`             | Panjang jendela smoothing (Savitzky–Golay).      |
| `-spoly`, `--smooth_poly`      |           | `int`     | `3`             | Orde polinomial smoothing.                       |
| `-sp`, `--smooth_p`            |           | `float`   | `0.5`           | Probabilitas apply smoothing.                    |

> Untuk detail lengkap:
>
> ```bash
> python augmentation.py --help
> ```

---

### 2. Menggunakan sebagai Modul Python

Anda dapat mengimpor dan menggunakan `AugmentationPipeline` langsung di dalam kode Python tanpa melalui CLI:

```python
from augmentation import AugmentationPipeline
import pandas as pd

# 1. Baca data
# Ganti dengan path dan nama kolom sesuai data Anda
df = pd.read_excel("example.xlsx")
X = df["Time"].values
Y = df["Intensity"].values

# 2. Atur parameter augmentasi (sama seperti argumen CLI)
params = {
    'X_shift':       {'range': (-0.1, 0.1), 'p': 0.5},
    'noise_std':     {'range': (0.005, 0.02), 'p': 0.5},
    'scale':         {'range': (0.95, 1.05), 'p': 0.5},
    'baseline':      {'range': (-0.01, 0.01), 'p': 0.5, 'nonlinear': False},
    'warp':          {'range': (-0.03, 0.03), 'p': 0.5, 'nonlinear': False},
    'smooth':        {'params': (7, 3), 'p': 0.5}
}
seed = 42

# 3. Inisialisasi pipeline dan generate augmentasi
pipeline = AugmentationPipeline(params, seed=seed)
n_samples = 5
augmented = pipeline.generate(X, Y, n_samples=n_samples)

# 4. Tambahkan hasil ke DataFrame
for i, y_aug in enumerate(augmented, 1):
    df[f"Augmentasi {i}"] = y_aug

# 5. Simpan hasil augmentasi ke file
# Bisa .xlsx, .csv, atau format lain yang didukung
df.to_excel("output.xlsx", index=False)
```

**Penjelasan**:

* `params`: dict konfigurasi transformasi, seperti opsi CLI.
* `pipeline.generate`: menghasilkan `n_samples` augmentasi sebagai `numpy.ndarray`.
* Hasilnya dapat langsung diproses atau disimpan.

---

## 📊 Contoh Data (`example.xlsx`)

| Time (detik) | Intensity |
| ------------ | --------- |
| 0.00         | 0.123     |
| 0.01         | 0.130     |
| …            | …         |

* **Time** → sumbu X
* **Intensity** → sinyal Y

Anda dapat menambahkan kolom Y lain dan memanggil `-y Col1,Col2`.

---

## 🛠️ Pengembangan & Kontribusi

1. Fork repo di GitHub:
   `https://github.com/Arifmaulanaazis/AugmentationPipeline`
2. Buat branch baru:

   ```bash
   git checkout -b feature/nama-fitur
   ```
3. Commit perubahan Anda & push ke branch:

   ```bash
   git commit -m "Deskripsi singkat"
   git push origin feature/nama-fitur
   ```
4. Buat Pull Request di GitHub.

Mohon sertakan deskripsi perubahan dan contoh penggunaan jika perlu.

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah **MIT License**.
Lihat [LICENSE](LICENSE) untuk detail.

---

## 📬 Kontak

Arif Maulana Azis — [@Arifmaulanaazis](https://github.com/Arifmaulanaazis)
Email: [titandigitalsoft@gmail.com](mailto:titandigitalsoft@gmail.com)
