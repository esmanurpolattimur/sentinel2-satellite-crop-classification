import subprocess
import sys
from pathlib import Path


# ============================================================
# FAZ 2 ÇALIŞTIRMA DOSYASI
# ============================================================

proje_klasoru = Path(__file__).resolve().parent

gerekli_dosyalar = [
    proje_klasoru / "data" / "phase1" / "tum_croptypes" / "data2016-2018.xlsx",
    proje_klasoru / "data" / "phase1" / "tum_croptypes" / "TestData.xlsx",
]

eksik_dosyalar = []

for dosya in gerekli_dosyalar:
    if not dosya.is_file():
        eksik_dosyalar.append(dosya)

if len(eksik_dosyalar) > 0:
    print("Eksik veri dosyaları:")
    for dosya in eksik_dosyalar:
        print(" -", dosya.relative_to(proje_klasoru))
    raise SystemExit(1)

print("TUM CropTypes model kodu çalıştırılıyor...")

subprocess.run(
    [sys.executable, str(proje_klasoru / "analysis" / "phase2_models.py")],
    cwd=proje_klasoru,
    check=True,
)

print("Faz 2 temel modelleri tamamlandı.")
