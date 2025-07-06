import os
import sys
import random
import argparse
import importlib
import subprocess
from types import SimpleNamespace


def auto_install_and_import(packages: dict):
    """
    Pastikan modul-modul di `packages` terinstal dan di-import.

    packages: dict dengan key = modul path (string yang dipakai di import),
              value = nama paket di pip.
    Contoh:
      {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy.signal": "scipy",
        "scipy.interpolate": "scipy",
      }
    """
    imported_modules = {}
    for module_path, pip_name in packages.items():
        try:
            # Coba import modul
            module = importlib.import_module(module_path)
        except ImportError:
            # Jika gagal, install via pip
            print(f"Module '{module_path}' tidak ditemukan. Menginstal '{pip_name}'…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            module = importlib.import_module(module_path)
        imported_modules[module_path] = module
    return imported_modules


# Dict untuk memastikan modul-modul yang diperlukan terinstal
required = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy.signal": "scipy",
    "scipy.interpolate": "scipy",
}

# Mengimpor modul-modul yang diperlukan, menginstal jika belum ada
mods = auto_install_and_import(required)
# Menggunakan modul-modul yang sudah diimpor
np = mods["numpy"]
pd = mods["pandas"]
signal = mods["scipy.signal"]
interp1d = mods["scipy.interpolate"].interp1d


class AugmentationPipeline:
    """
    Kelas untuk melakukan augmentasi data sinyal (misalnya kromatografi) dengan
    berbagai transformasi probabilistik dan non-linear.\
    Parameter augmentasi diatur sekali saat inisialisasi, dan method augment()\
    dapat dipanggil berulang kali untuk menghasilkan sampel baru.
    """

    def __init__(self, params: dict, seed: int = None):
        """
        Inisialisasi pipeline augmentasi.

        Argumen:
            params (dict): Dict konfigurasi augmentasi. Contoh format:
                {
                    'X_shift':       {'range': (min, max), 'p': prob},
                    'noise_std':     {'range': (min, max), 'p': prob},
                    'scale':         {'range': (min, max), 'p': prob},
                    'baseline':      {'range': (min, max), 'p': prob, 'nonlinear': bool},
                    'warp':          {'range': (min, max), 'p': prob, 'nonlinear': bool},
                    'smooth':        {'params': (window_length, polyorder), 'p': prob}
                }
            seed (int, optional): Nilai seed untuk reproducibility. Jika None, random tidak di-seed.
        """
        self.params = params
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _add_gaussian_noise(self, Y: np.ndarray, noise_std_range: tuple) -> np.ndarray:
        """
        Tambahkan noise Gaussian acak ke sinyal intensitas Y.

        Gaussian noise (white noise) adalah noise yang mengikuti distribusi normal (Gaussian),
        sering digunakan untuk mensimulasikan ketidaksempurnaan pengukuran atau gangguan acak
        di data eksperimen. Dengan menambahkan noise, model ML dapat belajar menjadi lebih tahan
        terhadap variasi nyata pada data.

        Argumen:
            Y (np.ndarray): Vektor satu dimensi intensitas asli.
            noise_std_range (tuple(float, float)): Rentang nilai deviasi standar (min, max)
                untuk noise yang dihasilkan.

        Returns:
            np.ndarray: Vektor intensitas baru dengan noise Gaussian.

        Perilaku terhadap argumen:
            - Jika noise_std_range mendekati 0, noise yang ditambahkan sangat kecil,
              sehingga hasil hampir identik dengan sinyal asli.
            - Jika noise_std_range besar, noise dapat mendominasi sinyal asli,
              menutupi pola utama pada data.
        """
        std = np.random.uniform(*noise_std_range)
        noise = np.random.normal(0, std, size=Y.shape)
        return Y + noise

    def _shift_X(self, X: np.ndarray, shift_range: tuple) -> np.ndarray:
        """
        Geser sumbu X secara linier.

        Transformasi ini digunakan untuk mensimulasikan variasi waktu retensi, Suhu pada
        instrumen kromatografi, DSC, misalnya pergeseran kecil karena perubahan suhu atau
        kondisi kolom.

        Argumen:
            X (np.ndarray): Vektor satu dimensi sumbu X asli.
            shift_range (tuple(float, float)): Rentang (min, max) pergeseran X.

        Returns:
            np.ndarray: Vektor X yang sudah digeser.

        Perilaku:
            - Pergeseran positif memajukan X deteksi puncak.
            - Pergeseran negatif memundurkan X stamp puncak.
            - Jika rentang kecil, efek geser minimal.
        """
        dt = np.random.uniform(*shift_range)
        return X + dt

    def _scale_intensity(self, Y: np.ndarray, scale_range: tuple) -> np.ndarray:
        """
        Skalakan intensitas Y secara proporsional.

        Digunakan untuk mensimulasikan variasi sensitivitas detektor atau jumlah sampel.

        Argumen:
            Y (np.ndarray): Vektor intensitas asli.
            scale_range (tuple(float, float)): Rentang faktor skala (min, max).

        Returns:
            np.ndarray: Vektor intensitas yang sudah diskalakan.

        Perilaku:
            - Faktor >1 meningkatkan amplitudo puncak.
            - Faktor <1 menurunkan amplitudo.
            - Faktor sangat besar atau kecil dapat membuat sinyal terlalu dominan
              atau terlalu lemah.
        """
        factor = np.random.uniform(*scale_range)
        return Y * factor

    def _baseline_drift(self, Y: np.ndarray, drift_range: tuple) -> np.ndarray:
        """
        Tambahkan baseline drift linier eksponensial pada sinyal Y.

        Drift baseline menggambarkan perubahan perlahan-lahan pada sinyal dasar
        yang sering terjadi akibat gradien pelarut atau kondisi instrumen.

        Argumen:
            Y (np.ndarray): Vektor intensitas asli.
            drift_range (tuple(float, float)): Rentang total drift yang dihasilkan.

        Returns:
            np.ndarray: Sinyal dengan drift baseline.

        Perilaku:
            - Drift positif menaikkan baseline seiring waktu.
            - Drift negatif menurunkan baseline.
            - Bentuk drift mengikuti eksponensial ter-skala.
        """
        total_drift = np.random.uniform(*drift_range)
        # Menghasilkan drift eksponensial ter-skala antara 0 dan total_drift
        drift = total_drift * (np.exp(np.linspace(0, 1, num=Y.shape[0])) - 1) / (np.e - 1)
        return Y + drift

    def _nonlinear_baseline_drift(self, Y: np.ndarray, drift_range: tuple) -> np.ndarray:
        """
        Tambahkan baseline drift non-linear acak pada sinyal Y.

        Metode ini menggunakan pergeseran acak berdasarkan distribusi normal
        terakumulasi (random walk) untuk mensimulasikan variasi baseline yang tidak
        teratur.

        Argumen:
            Y (np.ndarray): Vektor intensitas asli.
            drift_range (tuple(float, float)): Rentang maksimum amplitude drift.

        Returns:
            np.ndarray: Sinyal dengan non-linear baseline drift.

        Perilaku:
            - Semakin besar rentang, semakin curam fluktuasi baseline.
            - Jika rentang kecil, baseline relatif stabil.
        """
        step_scale = (drift_range[1] - drift_range[0]) / 10
        steps = np.random.normal(0, step_scale, size=Y.shape)
        drift = np.cumsum(steps)
        # Normalisasi drift hingga amplitude rentang yang diinginkan
        drift = drift / np.max(np.abs(drift)) * np.random.uniform(*drift_range)
        return Y + drift

    def _spectral_warp(self, X: np.ndarray, Y: np.ndarray, warp_range: tuple) -> tuple:
        """
        Lakukan spectral warp linier pada sumbu X.

        Digunakan untuk mensimulasikan perubahan kecepatan laju gradien atau
        variasi resolusi, dengan cara memampatkan atau meregangkan sumbu waktu.

        Argumen:
            X (np.ndarray): Vektor waktu asli.
            Y (np.ndarray): Vektor intensitas asli.
            warp_range (tuple(float, float)): Rentang faktor warp relatif.

        Returns:
            tuple(np.ndarray, np.ndarray): (X_asli, Y_interp), di mana Y_interp
            adalah intensitas yang diinterpolasi kembali pada X_asli.

        Perilaku:
            - Faktor >1 mempercepat time scale (kompresi).
            - Faktor <1 memperlambat (peregangan).
        """
        factor = 1 + np.random.uniform(*warp_range)
        X_warped = X * factor
        interp = interp1d(X_warped, Y, kind='linear', bounds_error=False, fill_value='extrapolate')
        return X, interp(X)

    def _nonlinear_spectral_warp(
        self, X: np.ndarray, Y: np.ndarray,
        warp_range: tuple, n_control_points: int = 5
    ) -> tuple:
        """
        Lakukan spectral warp non-linear menggunakan beberapa titik kontrol.

        Titik kontrol dipilih merata pada rentang X, kemudian diberi
        perpindahan acak untuk membentuk fungsi warp non-linear.

        Argumen:
            X (np.ndarray): Vektor waktu asli.
            Y (np.ndarray): Vektor intensitas asli.
            warp_range (tuple(float, float)): Rentang deformasi relatif.
            n_control_points (int): Jumlah titik kontrol untuk interpolasi warp.

        Returns:
            tuple(np.ndarray, np.ndarray): (X_asli, Y_interp) setelah warp.
        """
        min_x, max_x = X.min(), X.max()
        cp_x = np.linspace(min_x, max_x, n_control_points)
        cp_disp = np.random.uniform(*warp_range, size=n_control_points) * (max_x - min_x)
        drift = np.interp(X, cp_x, cp_disp)
        X_warped = X + drift
        interp = interp1d(X_warped, Y, kind='linear', bounds_error=False, fill_value='extrapolate')
        return X, interp(X)

    def _smooth_signal(self, Y: np.ndarray, window_length: int = 7, polyorder: int = 3) -> np.ndarray:
        """
        Haluskan sinyal menggunakan filter Savitzky–Golay.

        Digunakan untuk meredam fluktuasi frekuensi tinggi tanpa mengubah
        posisi puncak secara signifikan.

        Argumen:
            Y (np.ndarray): Vektor intensitas asli.
            window_length (int): Panjang jendela smoothing (harus ganjil).
            polyorder (int): Orde polinomial untuk fitting pada jendela.

        Returns:
            np.ndarray: Sinyal halus.

        Perilaku:
            - Window besar meredam noise lebih kuat, tapi dapat menyamaratakan
              fitur penting.
            - Polyorder rendah menghasilkan smoothing lebih lurus.
        """
        wl = min(window_length if window_length % 2 == 1 else window_length + 1,
                 Y.shape[0] - (1 - Y.shape[0] % 2))
        return signal.savgol_filter(Y, wl, polyorder)

    def augment(self, X: np.ndarray, Y: np.ndarray) -> tuple:
        """
        Terapkan satu iterasi augmentasi data berdasarkan probabilitas pada setiap transform.

        Argumen:
            X (np.ndarray): Vektor waktu/orisinal.
            Y (np.ndarray): Vektor intensitas/orisinal.

        Returns:
            tuple(np.ndarray, np.ndarray): Pasangan (X_aug, Y_aug).
        """
        X_aug, Y_aug = X.copy(), Y.copy()
        # Pilih transform yang akan dipakai
        transforms = [name for name, cfg in self.params.items()
                      if random.random() < cfg.get('p', 1.0)]
        random.shuffle(transforms)

        for name in transforms:
            cfg = self.params[name]
            if name == 'X_shift':
                X_aug = self._shift_X(X_aug, cfg['range'])
            elif name == 'noise_std':
                Y_aug = self._add_gaussian_noise(Y_aug, cfg['range'])
            elif name == 'scale':
                Y_aug = self._scale_intensity(Y_aug, cfg['range'])
            elif name == 'baseline':
                if cfg.get('nonlinear', False):
                    Y_aug = self._nonlinear_baseline_drift(Y_aug, cfg['range'])
                else:
                    Y_aug = self._baseline_drift(Y_aug, cfg['range'])
            elif name == 'warp':
                if cfg.get('nonlinear', False):
                    X_aug, Y_aug = self._nonlinear_spectral_warp(X_aug, Y_aug, cfg['range'])
                else:
                    X_aug, Y_aug = self._spectral_warp(X_aug, Y_aug, cfg['range'])
            elif name == 'smooth':
                wl, po = cfg['params']
                Y_aug = self._smooth_signal(Y_aug, wl, po)

        return X_aug, Y_aug

    def generate(self, X: np.ndarray, Y: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Menghasilkan beberapa sampel augmentasi dari sinyal input.

        Argumen:
            X (np.ndarray): Vektor waktu/orisinal.
            Y (np.ndarray): Vektor intensitas/orisinal.
            n_samples (int): Jumlah sampel augmentasi yang diinginkan.

        Returns:
            np.ndarray: Array 2D berukuran (n_samples, len(Y)),
            di mana setiap baris adalah hasil augmentasi.

        Perilaku:
            - Setiap iterasi memanggil method augment() sekali.
            - Hasil urutan augment berbeda jika seed tidak diset.
        """
        results = []
        for _ in range(n_samples):
            _, y_aug = self.augment(X, Y)
            results.append(y_aug)
        return np.array(results)





def parse_args():
    """
    Fungsi untuk mengurai argumen command line.
    Digunakan untuk menjalankan pipeline augmentasi dari command line.
    """
    parser = argparse.ArgumentParser(
        description='Hasilkan data augmentasi dari data sinyal masukan (misalnya, kromatografi) menggunakan transformasi.'
    )
    # Argumen utama
    parser.add_argument( '-i',    '--input',               type=str,              required=True,          help='Path file input (.csv atau .xlsx)' )
    parser.add_argument( '-x',    '--input_x',             type=str,              required=True,          help='Nama kolom sumbu X' )
    parser.add_argument( '-y',    '--input_y',             type=str,              required=True,          help='Nama kolom Y, pisahkan koma jika lebih dari satu' )
    parser.add_argument( '-o',    '--output',              type=str,              default='output.xlsx',  help='Path file output (.xlsx, .csv, .npz)' )
    parser.add_argument( '-n',    '--n_samples',           type=int,              default=1,              help='Jumlah sampel augmentasi' )
    parser.add_argument( '-s',    '--seed',                type=int,              default=None,           help='Seed untuk reproducibility' )
    parser.add_argument( '-ka',   '--kolom_augmentasi',    type=str,              default='Augmentasi',   help='Prefix untuk nama kolom augmentasi' )
    # Argumen parameter augmentasi
    parser.add_argument( '-tr',   '--X_shift_range',        type=float, nargs=2,   default=[-0.1,0.1],     help='Rentang shift X' )
    parser.add_argument( '-tp',   '--X_shift_p',            type=float,             default=0.5,            help='Probabilitas apply X shift' )
    parser.add_argument( '-nr',   '--noise_std_range',      type=float, nargs=2,   default=[0.005,0.02],   help='Rentang std noise' )
    parser.add_argument( '-np',   '--noise_std_p',          type=float,             default=0.5,            help='Probabilitas apply noise' )
    parser.add_argument( '-scr',  '--scale_range',          type=float, nargs=2,   default=[0.95,1.05],    help='Rentang skala intensitas' )
    parser.add_argument( '-scp',  '--scale_p',              type=float,             default=0.5,            help='Probabilitas apply scale' )
    parser.add_argument( '-br',   '--baseline_range',       type=float, nargs=2,   default=[-0.01,0.01],   help='Rentang baseline drift' )
    parser.add_argument( '-bp',   '--baseline_p',           type=float,             default=0.5,            help='Probabilitas apply baseline drift' )
    parser.add_argument( '-bln',  '--baseline_nonlinear',    action='store_true',    help='Gunakan nonlinear baseline drift' )
    parser.add_argument( '-wr',   '--warp_range',           type=float, nargs=2,   default=[-0.03,0.03],   help='Rentang spectral warp' )
    parser.add_argument( '-wp',   '--warp_p',               type=float,             default=0.5,            help='Probabilitas apply spectral warp' )
    parser.add_argument( '-wnl',  '--warp_nonlinear',        action='store_true',    help='Gunakan nonlinear spectral warp' )
    parser.add_argument( '-sw',   '--smooth_window',        type=int,               default=7,              help='Window length untuk smoothing' )
    parser.add_argument( '-spoly','--smooth_poly',          type=int,               default=3,              help='Polyorder untuk smoothing' )
    parser.add_argument( '-sp',   '--smooth_p',             type=float,             default=0.5,            help='Probabilitas apply smoothing' )

    return parser.parse_args()

def run(args: argparse.Namespace = None, **kwargs):
    """
    Jika dipanggil tanpa args dan kwargs, baca dari CLI.
    Jika dipanggil dengan args=Namespace atau kwargs, gunakan nilai tersebut.
    """
    # 1 Tentukan sumber argumen
    if args is None and not kwargs:
        args = parse_args()
    elif not isinstance(args, SimpleNamespace):
        # Bungkus kwargs menjadi Namespace
        args = SimpleNamespace(**kwargs)

    # 2 Bangun dict params untuk pipeline
    params = {
        'X_shift':    {'range': tuple(args.X_shift_range),    'p': args.X_shift_p},
        'noise_std':  {'range': tuple(args.noise_std_range),  'p': args.noise_std_p},
        'scale':      {'range': tuple(args.scale_range),      'p': args.scale_p},
        'baseline':   {'range': tuple(args.baseline_range),   'p': args.baseline_p, 'nonlinear': args.baseline_nonlinear},
        'warp':       {'range': tuple(args.warp_range),       'p': args.warp_p,     'nonlinear': args.warp_nonlinear},
        'smooth':     {'params': (args.smooth_window, args.smooth_poly), 'p': args.smooth_p}
    }

    # 3 Baca data input
    ext_in = os.path.splitext(args.input)[1].lower()
    if ext_in in ['.xlsx', '.xls']:
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)

    X = df[args.input_x].values
    y_cols = [col.strip() for col in args.input_y.split(',')]
    df_out = df[[args.input_x] + y_cols].copy()

    # 4 Inisialisasi pipeline dan generate augmentasi
    pipeline = AugmentationPipeline(params, seed=args.seed)
    for i in range(1, args.n_samples + 1):
        y_orig = df[y_cols[(i - 1) % len(y_cols)]].values
        aug = pipeline.generate(X, y_orig, n_samples=1)[0]
        df_out[f"{args.kolom_augmentasi} {i}"] = aug

    # 5 Simpan output sesuai ekstensi
    ext_out = os.path.splitext(args.output)[1].lower()
    if ext_out in ['.xlsx', '.xls']:
        df_out.to_excel(args.output, index=False)
    elif ext_out == '.csv':
        df_out.to_csv(args.output, index=False)
    else:
        # .npz atau format lain
        np.savez(args.output,
                 X=X,
                 **{col: df_out[col].values for col in df_out.columns if col != args.input_x})
        
    print(f"Output disimpan ke {args.output} dengan total {len(df_out.columns) - 1} kolom sampel.")
    # 6 Kembalikan DataFrame output
    return df_out


if __name__ == '__main__':
    # Panggil fungsi run()
    run()
