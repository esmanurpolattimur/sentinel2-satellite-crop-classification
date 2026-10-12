import subprocess
import sys
from pathlib import Path


# ============================================================
# FAZ 1 ÇALIŞTIRMA DOSYASI
# ============================================================

proje_klasoru = Path(__file__).resolve().parent

gerekli_dosyalar = [
    "data/phase1/breizhcrops/classmapping.csv",
    "data/phase1/breizhcrops/frh01.csv",
    "data/phase1/breizhcrops/frh02.csv",
    "data/phase1/breizhcrops/frh03.csv",
    "data/phase1/breizhcrops/frh04.csv",
    "data/phase1/tum_croptypes/data2016-2018.xlsx",
    "data/phase1/tum_croptypes/TestData.xlsx",
    "breizhcrops_frh04_evidence/6082638.csv",
]


def dosyalari_kontrol_et():
    eksik_dosyalar = []

    for dosya in gerekli_dosyalar:
        tam_yol = proje_klasoru / dosya
        if not tam_yol.is_file():
            eksik_dosyalar.append(dosya)

    if len(eksik_dosyalar) > 0:
        print("Eksik veri dosyaları:")
        for dosya in eksik_dosyalar:
            print(" -", dosya)
        return False

    print("Bütün veri dosyaları bulundu.")
    return True


def python_dosyasi_calistir(dosya):
    print("\nÇalıştırılıyor:", dosya)
    subprocess.run(
        [sys.executable, str(proje_klasoru / dosya)],
        cwd=proje_klasoru,
        check=True,
    )


# Komut verilmezse analiz ve rapor birlikte çalışır.
if len(sys.argv) > 1:
    secim = sys.argv[1]
else:
    secim = "--all"

gecerli_secimler = ["--check", "--eda-only", "--report-only", "--all"]

if secim not in gecerli_secimler:
    print("Kullanım: python run_phase1.py [--check|--eda-only|--report-only|--all]")
    raise SystemExit(1)

if secim == "--check":
    if not dosyalari_kontrol_et():
        raise SystemExit(1)

elif secim == "--report-only":
    # Rapor, daha önce oluşturulan özet ve grafiklerden tekrar üretilebilir.
    ozet = proje_klasoru / "output" / "phase1" / "eda_summary.json"
    if not ozet.is_file():
        print("Önce veri analizini çalıştırmanız gerekiyor.")
        raise SystemExit(1)
    python_dosyasi_calistir("analysis/build_phase1_report.py")

else:
    if not dosyalari_kontrol_et():
        raise SystemExit(1)

    python_dosyasi_calistir("analysis/phase1_eda.py")

    if secim == "--all":
        python_dosyasi_calistir("analysis/build_phase1_report.py")

print("\nİşlem tamamlandı.")
