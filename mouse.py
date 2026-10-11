"""
MOUSE TANGAN v9 - gerakkan kursor dengan jari lewat kamera laptop (gratis, offline).

Cara pakai:
  - Angkat TELUNJUK (jari lain menggulung)   -> kursor bergerak mengikuti gerakan tanganmu
  - GULUNG telunjuk sebentar lalu ANGKAT lagi -> KLIK KIRI (seperti mengetuk tombol)
        kursor otomatis terkunci selama telunjuk menggulung, jadi klik tepat sasaran
  - Angkat JARI TENGAH (bentuk V)            -> KLIK KANAN (sekali tiap jari tengah diangkat)
  - GENGGAM ERAT dan TAHAN (+- 0.5 detik)    -> mulai DRAG (tombol kiri ditahan),
                                                geser tangan untuk menyeret
  - BUKA genggaman (angkat 2 jari atau lebih) -> LEPAS drag
  - Telapak terbuka / tangan keluar kamera   -> istirahat, kursor diam
  - Klik dua kali = gulung-angkat-gulung-angkat dengan cepat

KELUAR (satu tangan, hitung mundur dengan jari):
  1. BUKA 5 JARI (telapak terbuka) dan tahan sebentar (+- 0.4 detik)
  2. Tutup jari SATU PER SATU, berurutan:
        kelingking -> manis -> tengah -> telunjuk -> jempol
     (seperti menghitung mundur 5-4-3-2-1; tiap jari berikutnya harus ditutup dalam 1.5 detik)
  3. Genggaman terakhir ditahan sebentar -> program menutup sendiri.
  Kalau urutannya salah atau terlalu lama, hitungan batal; buka telapak lagi untuk mengulang.
  Menutup tangan sekaligus (langsung mengepal) TIDAK akan keluar, jadi aman dari salah pencet.

BEDA KLIK vs DRAG: sama-sama menggulung telunjuk, bedanya lamanya.
  Gulung SEBENTAR (di bawah 0.4 detik) lalu angkat = klik.
  Genggam erat dan TAHAN (0.5 detik atau lebih) = drag.

MODE RELATIF (default), cara kerjanya seperti touchpad:
  - Yang dihitung adalah ARAH dan JARAK geseran tanganmu, bukan posisi di kamera,
    jadi tangan boleh dekat atau jauh dari laptop.
  - Kursor mengikuti pangkal telunjuk (bukan ujungnya), jadi menggulung telunjuk
    tidak menggeser kursor.
  - Kalau tangan sudah mentok, buka telapak (istirahat), geser kembali ke tengah,
    lalu angkat telunjuk lagi.

ANTI GETAR: posisi disaring dengan filter "One Euro". Saat tangan diam, kursor
benar-benar diam. Saat tangan bergerak cepat, filter melonggar supaya kursor tetap responsif.

Di jendela kamera ada angka "telunjuk", "kepal", dan "jempol" untuk membantu menyetel.
Tombol di jendela kamera:  M = ganti mode (relatif / absolut),  Q atau ESC = keluar

Persiapan (sekali saja):
    pip install opencv-python mediapipe pyautogui numpy

Jalankan:
    python mouse_tangan.py
"""

import math
import platform
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ---------------- pengaturan (silakan diubah) ----------------
KAMERA = 0                # 0 = kamera bawaan laptop, coba 1 kalau pakai webcam lain
LEBAR, TINGGI = 640, 480
KUALITAS_MODEL = 1        # 1 = lebih akurat & stabil (agak berat), 0 = lebih ringan tapi lebih goyang

# --- mode relatif ---
SENSITIVITAS = 0.8        # geser tangan sejauh 1 telapak = kursor bergerak 0.8 lebar layar
AKSELERASI = 1.0          # geser cepat = kursor makin jauh (0 = tanpa akselerasi)
ANTI_GETAR = 1.0          # makin KECIL = kursor makin diam saat tangan diam (coba 0.5 - 2.0)
KEJAR_GERAK = 15.0        # makin BESAR = kursor makin cepat menyusul saat tangan digeser (5 - 30)
AMBANG_DIAM = 0.012       # gerakan kecil di bawah ini diabaikan (satuan: panjang telapak/frame)

# --- mode absolut (cadangan, tekan M untuk pindah) ---
AREA_X = (0.20, 0.80)     # bagian kamera (kiri-kanan) yang dipetakan ke seluruh layar
AREA_Y = (0.10, 0.70)     # bagian kamera (atas-bawah) yang dipetakan ke seluruh layar
HALUS = 4                 # makin besar = kursor makin halus tapi agak lambat (3-8)

# --- klik kiri (telunjuk digulung sebentar lalu diangkat lagi) ---
GULUNG_DALAM = 1.25       # telunjuk harus digulung sampai angka "telunjuk" di bawah ini (cegah klik tak sengaja)
                          # besarkan kalau gulungan ringan tak terbaca, kecilkan kalau sering klik sendiri
TAP_MIN = 0.04            # detik: gulungan lebih singkat dari ini dianggap noise
TAP_MAKS = 0.40           # detik: gulungan lebih lama dari ini BUKAN klik lagi (jadi calon drag)
JEDA_KLIK = 0.2           # detik minimal antar klik

# --- drag (genggam erat dan tahan) ---
DRAG_AKTIF = True         # False = fitur drag dimatikan total
DRAG_TAHAN = 0.5          # detik: genggam harus ditahan segini lama baru drag mulai (harus > TAP_MAKS)
KEPAL_MAX = 1.0           # genggam hanya sah kalau SEMUA ujung jari lebih dekat dari ini ke pergelangan
                          # (lihat angka "kepal" di jendela kamera; besarkan kalau genggaman erat tak terbaca)
JEDA_DRAG_SETELAH_KLIK = 0.6   # detik: setelah klik, drag diblokir dulu (cegah drag tak sengaja)
FRAME_LEPAS = 3           # tangan harus terbuka segini frame baru drag dilepas
FRAME_HILANG = 10         # kalau tangan hilang dari kamera segini frame saat drag, drag dilepas otomatis

# --- umum ---
SENSITIF_JARI = 1.2       # makin kecil = jari lebih mudah dianggap 'terangkat'

# --- keluar: telapak terbuka lalu tutup jari berurutan kelingking -> jempol (pengaturan TERPISAH) ---
KELUAR_AKTIF = True       # False = gestur keluar mati (tetap bisa keluar dengan tombol Q / ESC)
KELUAR_PAKAI_JEMPOL = True   # False = jempol diabaikan (urutan 4 jari saja), berguna kalau jempol sulit terbaca
KELUAR_BUKA_TAHAN = 0.4   # detik: telapak (semua jari) harus terbuka segini lama sebelum hitungan boleh dimulai
KELUAR_LANGKAH_MAKS = 1.5 # detik: batas waktu antar lipatan jari; lebih lama dari ini hitungan batal
KELUAR_TAHAN_AKHIR = 0.3  # detik: genggaman terakhir ditahan segini lama baru keluar (harus < DRAG_TAHAN)
JEMPOL_BUKA = 0.55        # jempol dianggap terbuka kalau angka "jempol" di jendela kamera di atas ini
                          # (besarkan kalau jempol sering dikira terbuka, kecilkan kalau jempol terbuka tak terbaca)

pyautogui.FAILSAFE = False   # supaya kursor boleh sampai pojok layar
pyautogui.PAUSE = 0
LAYAR_W, LAYAR_H = pyautogui.size()

URL_MODEL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

NAMA_URUTAN = ["kelingking", "manis", "tengah", "telunjuk", "jempol"]


# ---------------- filter anti getar (One Euro Filter) ----------------
class SatuEuro:
    """Penyaring yang kuat saat gerakan lambat (hilangkan getar) dan longgar saat gerakan cepat."""

    def __init__(self, min_cutoff=1.0, beta=15.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-3)
        d = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * d + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


# ---------------- gestur keluar: hitung mundur dengan jari ----------------
class GestKeluar:
    """Telapak terbuka, lalu jari ditutup berurutan kelingking -> manis -> tengah -> telunjuk -> jempol."""

    def __init__(self):
        self.total = 5 if KELUAR_PAKAI_JEMPOL else 4
        self.reset()

    def reset(self):
        self.tahap = 0          # sudah berapa jari yang ditutup (0 = belum mulai)
        self.siap = False       # telapak sudah terbuka cukup lama
        self.t_buka = None
        self.t_langkah = None
        self.t_akhir = None

    def update(self, naik, sekarang):
        """naik = [kelingking, manis, tengah, telunjuk, jempol], True kalau terangkat.
        Mengembalikan True kalau gestur selesai (waktunya keluar)."""
        naik = naik[:self.total]
        # k = jumlah jari berurutan dari kelingking yang tertutup; None kalau polanya bukan urutan yang benar
        k = 0
        while k < self.total and not naik[k]:
            k += 1
        if any(not v for v in naik[k:]):
            k = None

        if self.tahap == 0:
            if k == 0:                                   # telapak terbuka
                if self.t_buka is None:
                    self.t_buka = sekarang
                self.siap = sekarang - self.t_buka >= KELUAR_BUKA_TAHAN
            elif k == 1 and self.siap:                   # kelingking ditutup -> mulai hitung mundur
                self.tahap = 1
                self.t_langkah = sekarang
            else:
                self.t_buka, self.siap = None, False
            return False

        if k == self.tahap:                              # masih di langkah yang sama
            if self.tahap == self.total:                 # genggaman akhir: tahan sebentar
                if self.t_akhir is None:
                    self.t_akhir = sekarang
                return sekarang - self.t_akhir >= KELUAR_TAHAN_AKHIR
            if sekarang - self.t_langkah > KELUAR_LANGKAH_MAKS:
                self.reset()                             # terlalu lama, batal
        elif k == self.tahap + 1:                        # jari berikutnya ditutup
            self.tahap = k
            self.t_langkah = sekarang
        else:                                            # urutan salah / jari dibuka lagi -> batal
            self.reset()
        return False


# ---------------- deteksi tangan ----------------
class Detektor:
    """Mengembalikan 21 titik tangan (x, y dalam 0..1), atau None kalau tangan tidak terlihat."""

    def __init__(self):
        self.mode = None
        try:  # cara lama (paling simpel)
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=1,
                model_complexity=KUALITAS_MODEL,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )
            self.mode = "solutions"
        except AttributeError:  # versi mediapipe baru tanpa mp.solutions
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            path = self._siapkan_model()
            opsi = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=path),
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(opsi)
            self.mode = "tasks"

    @staticmethod
    def _siapkan_model():
        path = Path(__file__).with_name("hand_landmarker.task")
        if not path.exists():
            print("Mengunduh model tangan (sekali saja, sekitar 8 MB)...")
            urllib.request.urlretrieve(URL_MODEL, path)
        return str(path)

    def proses(self, rgb):
        if self.mode == "solutions":
            hasil = self.hands.process(rgb)
            if hasil.multi_hand_landmarks:
                return [(p.x, p.y) for p in hasil.multi_hand_landmarks[0].landmark]
            return None
        gambar = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hasil = self.landmarker.detect(gambar)
        if hasil.hand_landmarks:
            return [(p.x, p.y) for p in hasil.hand_landmarks[0]]
        return None


def jarak(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def jari_naik(lm, ujung, sendi):
    """Jari dianggap terangkat kalau ujungnya jauh lebih jauh dari pergelangan dibanding sendinya."""
    return jarak(lm[0], lm[ujung]) > SENSITIF_JARI * jarak(lm[0], lm[sendi])


# ---------------- program utama ----------------
def main():
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(KAMERA, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(KAMERA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LEBAR)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TINGGI)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera, "
              "atau ubah KAMERA di bagian atas file.")
        return

    detektor = Detektor()
    gest = GestKeluar()
    relatif = True
    cx, cy = (float(v) for v in pyautogui.position())
    prev_tip = None         # posisi tersaring frame sebelumnya (None = awal sesi gerak)
    filt = None             # pasangan filter (x, y)
    ukuran_halus = 0.2      # ukuran telapak tangan di kamera (sudah dihaluskan)
    frame_kanan = 0
    frame_lepas = 0
    frame_hilang = 0
    sudah_kanan = False
    sedang_drag = False
    terakhir_klik = 0.0
    kondisi_lalu = None
    waktu_gulung = None     # kapan telunjuk mulai digulung
    gulung_dalam = False    # sudah cukup dalam menggulungnya?
    kilat_teks, kilat_sampai = "", 0.0   # tulisan 'KLIK!' sebentar di preview
    print("Mouse tangan aktif. M = ganti mode, Q/ESC = keluar (di jendela kamera).")
    if KELUAR_AKTIF:
        print("Keluar juga bisa: buka 5 jari, lalu tutup berurutan kelingking -> manis -> tengah -> telunjuk%s."
              % (" -> jempol" if KELUAR_PAKAI_JEMPOL else ""))

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Gagal membaca kamera.")
            break
        frame = cv2.flip(frame, 1)  # cermin, supaya gerak kanan = kursor ke kanan
        tinggi, lebar = frame.shape[:2]
        lm = detektor.proses(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        sekarang = time.time()
        ext_telunjuk = None
        ujung_terjauh = None
        nilai_jempol = None

        if lm is None:
            status = "TANGAN TIDAK TERLIHAT"
            kondisi_lalu = None
            waktu_gulung = None
            frame_kanan = frame_lepas = 0
            sudah_kanan = False
            prev_tip = None
            gest.reset()
            if sedang_drag:
                frame_hilang += 1
                if frame_hilang >= FRAME_HILANG:
                    pyautogui.mouseUp()          # jangan sampai tombol mouse tertahan selamanya
                    sedang_drag = False
                    status = "TANGAN HILANG - DRAG DILEPAS"
                else:
                    status = "DRAG (tangan hilang...)"
        else:
            frame_hilang = 0
            telunjuk = jari_naik(lm, 8, 6)
            tengah = jari_naik(lm, 12, 10)
            manis = jari_naik(lm, 16, 14)
            kelingking = jari_naik(lm, 20, 18)
            jumlah_naik = sum((telunjuk, tengah, manis, kelingking))

            ukuran = max(jarak(lm[0], lm[9]), 0.05)               # panjang telapak di kamera
            ext_telunjuk = jarak(lm[0], lm[8]) / ukuran            # makin kecil = telunjuk makin tergulung
            ujung_terjauh = max(jarak(lm[0], lm[i]) for i in (8, 12, 16, 20)) / ukuran
            kepal_ketat = ujung_terjauh < KEPAL_MAX                # genggaman sungguhan
            nilai_jempol = jarak(lm[4], lm[5]) / ukuran            # makin besar = jempol makin terbuka
            jempol = nilai_jempol > JEMPOL_BUKA

            # --- gestur KELUAR: telapak terbuka lalu jari ditutup berurutan kelingking -> jempol ---
            if KELUAR_AKTIF and not sedang_drag:
                if gest.update([kelingking, manis, tengah, telunjuk, jempol], sekarang):
                    break                      # keluar; tombol mouse aman (tidak sedang drag)
            else:
                gest.reset()

            # --- tentukan kondisi tangan saat ini ---
            # (selama hitungan mundur berjalan, bentuk V sementara tidak dianggap klik kanan)
            if sedang_drag:
                kondisi = "LEPAS" if jumlah_naik >= 2 else "DRAG"
            elif tengah and not manis and not kelingking and gest.tahap == 0:
                kondisi = "KLIK KANAN"
            elif not telunjuk and not tengah:
                kondisi = "GULUNG"
            elif telunjuk and not tengah:
                kondisi = "GERAK"
            else:
                kondisi = "ISTIRAHAT"

            # --- catat awal gulungan telunjuk ---
            if kondisi == "GULUNG":
                if kondisi_lalu != "GULUNG":
                    waktu_gulung = sekarang
                    gulung_dalam = False
                if ext_telunjuk < GULUNG_DALAM:
                    gulung_dalam = True
            elif kondisi != "GERAK":
                waktu_gulung = None

            # --- KLIK KIRI: telunjuk diangkat lagi setelah digulung sebentar ---
            if (kondisi == "GERAK" and kondisi_lalu == "GULUNG" and gulung_dalam
                    and waktu_gulung is not None
                    and TAP_MIN <= sekarang - waktu_gulung <= TAP_MAKS
                    and sekarang - terakhir_klik > JEDA_KLIK):
                pyautogui.click()
                terakhir_klik = sekarang
                kilat_teks, kilat_sampai = "KLIK!", sekarang + 0.35
            if kondisi == "GERAK":
                waktu_gulung = None
                gulung_dalam = False

            kondisi_lalu = kondisi

            # reset penghitung yang tidak sedang dipakai
            if kondisi != "KLIK KANAN":
                frame_kanan, sudah_kanan = 0, False
            if kondisi != "LEPAS":
                frame_lepas = 0
            if kondisi not in ("GERAK", "DRAG"):
                prev_tip = None

            status = kondisi

            if kondisi in ("GERAK", "DRAG"):
                # --- mode GERAK / DRAG: kursor mengikuti pangkal telunjuk (stabil walau jari menggulung) ---
                tip = lm[5]
                if relatif:
                    if prev_tip is None:
                        # awal sesi gerak: sinkronkan dengan posisi kursor sebenarnya
                        cx, cy = (float(v) for v in pyautogui.position())
                        filt = (SatuEuro(ANTI_GETAR, KEJAR_GERAK), SatuEuro(ANTI_GETAR, KEJAR_GERAK))
                        ukuran_halus = ukuran
                        prev_tip = (filt[0](tip[0], sekarang), filt[1](tip[1], sekarang))
                    else:
                        ukuran_halus += 0.1 * (ukuran - ukuran_halus)
                        halus = (filt[0](tip[0], sekarang), filt[1](tip[1], sekarang))
                        dx = (halus[0] - prev_tip[0]) / ukuran_halus
                        dy = (halus[1] - prev_tip[1]) / ukuran_halus
                        v = (dx * dx + dy * dy) ** 0.5
                        if v > AMBANG_DIAM:
                            efektif = v - AMBANG_DIAM          # zona mati yang halus (tanpa lompatan)
                            faktor = efektif / v
                            kali = SENSITIVITAS * (1 + AKSELERASI * min(efektif, 0.5))
                            cx += dx * faktor * kali * LAYAR_W
                            cy += dy * faktor * kali * LAYAR_H
                        prev_tip = halus
                else:
                    target_x = np.interp(tip[0], AREA_X, (0, LAYAR_W))
                    target_y = np.interp(tip[1], AREA_Y, (0, LAYAR_H))
                    cx += (target_x - cx) / HALUS
                    cy += (target_y - cy) / HALUS
                cx = float(np.clip(cx, 1, LAYAR_W - 2))
                cy = float(np.clip(cy, 1, LAYAR_H - 2))
                pyautogui.moveTo(int(cx), int(cy))

            elif kondisi == "GULUNG":
                # --- telunjuk menggulung: kursor dikunci. Ditahan lama + genggam erat = mulai drag ---
                # (drag tidak dimulai selama hitungan mundur keluar sedang berjalan)
                durasi = sekarang - waktu_gulung
                status = "GULUNG..."
                if (DRAG_AKTIF and kepal_ketat and durasi >= DRAG_TAHAN and gest.tahap == 0
                        and sekarang - terakhir_klik > JEDA_DRAG_SETELAH_KLIK):
                    pyautogui.mouseDown()
                    sedang_drag = True
                    waktu_gulung = None
                    status = "DRAG"
                elif durasi > TAP_MAKS:
                    status = "TAHAN... (genggam erat = drag)" if DRAG_AKTIF else "GULUNG (terlalu lama)"

            elif kondisi == "KLIK KANAN":
                # --- KLIK KANAN (kursor dikunci, klik sekali tiap jari tengah diangkat) ---
                frame_kanan += 1
                status = "KLIK KANAN!" if sudah_kanan else "KLIK KANAN..."
                if (frame_kanan >= 2 and not sudah_kanan
                        and sekarang - terakhir_klik > JEDA_KLIK):
                    pyautogui.rightClick()
                    terakhir_klik = sekarang
                    sudah_kanan = True
                    status = "KLIK KANAN!"

            elif kondisi == "LEPAS":
                # --- tangan terbuka beberapa frame -> lepas drag ---
                frame_lepas += 1
                status = "MELEPAS..."
                if frame_lepas >= FRAME_LEPAS:
                    pyautogui.mouseUp()
                    sedang_drag = False
                    frame_lepas = 0
                    status = "DILEPAS"

            # gambar titik tangan di preview
            for x, y in lm:
                cv2.circle(frame, (int(x * lebar), int(y * tinggi)), 3, (0, 255, 0), -1)
            cv2.circle(frame, (int(lm[8][0] * lebar), int(lm[8][1] * tinggi)), 9, (255, 0, 0), 2)
            cv2.circle(frame, (int(lm[12][0] * lebar), int(lm[12][1] * tinggi)), 9, (0, 0, 255), 2)
            cv2.circle(frame, (int(lm[5][0] * lebar), int(lm[5][1] * tinggi)), 7, (0, 255, 255), 2)

        # kotak area hanya relevan di mode absolut
        if not relatif:
            cv2.rectangle(frame,
                          (int(AREA_X[0] * lebar), int(AREA_Y[0] * tinggi)),
                          (int(AREA_X[1] * lebar), int(AREA_Y[1] * tinggi)), (255, 255, 0), 2)
        if sekarang < kilat_sampai:
            status = kilat_teks
        if gest.tahap >= 1:
            status = "KELUAR %d/%d" % (gest.tahap, gest.total)
        warna_status = (0, 165, 255) if sedang_drag else (0, 255, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, warna_status, 2)
        if ext_telunjuk is not None:
            cv2.putText(frame, "telunjuk: %.2f (klik jika sempat < %.2f)" % (ext_telunjuk, GULUNG_DALAM),
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if ujung_terjauh is not None:
            cv2.putText(frame, "kepal: %.2f (drag jika < %.2f)" % (ujung_terjauh, KEPAL_MAX),
                        (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if nilai_jempol is not None and KELUAR_AKTIF and KELUAR_PAKAI_JEMPOL:
            cv2.putText(frame, "jempol: %.2f (terbuka jika > %.2f)" % (nilai_jempol, JEMPOL_BUKA),
                        (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if KELUAR_AKTIF and lm is not None:
            if gest.tahap == 0 and gest.siap:
                cv2.putText(frame, "keluar siap: tutup kelingking dulu", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            elif 1 <= gest.tahap < gest.total:
                cv2.putText(frame, "tutup berikutnya: " + NAMA_URUTAN[gest.tahap], (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            elif gest.tahap == gest.total:
                cv2.putText(frame, "tahan genggaman...", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(frame, "MODE: " + ("RELATIF" if relatif else "ABSOLUT") + "  (M = ganti)",
                    (10, tinggi - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("Mouse Tangan - tekan Q untuk keluar", frame)

        tombol = cv2.waitKey(1) & 0xFF
        if tombol in (ord("q"), 27):
            break
        if tombol == ord("m"):
            relatif = not relatif
            prev_tip = None

    if sedang_drag:
        pyautogui.mouseUp()   # pastikan tombol mouse dilepas saat keluar
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()