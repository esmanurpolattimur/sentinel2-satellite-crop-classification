import argparse
import subprocess
import sys
from pathlib import Path


# ============================================================
# BREIZHCROPS PİLOT ÇALIŞTIRMA DOSYASI
# ============================================================

proje_klasoru = Path(__file__).resolve().parent


def ayarlari_oku():
    parser = argparse.ArgumentParser(
        description="BreizhCrops pilot ön işleme ve Random Forest çalıştırır."
    )
    parser.add_argument(
        "--veri-klasoru",
        type=Path,
        default=Path(r"D:\Datasets\BreizhCrops\2017\L1C"),
    )
    parser.add_argument("--oran", type=float, default=0.10)
    parser.add_argument(
        "--hizli-test",
        action="store_true",
        help="Her bölge-sınıftan en fazla 2 parsel ile kodu dener",
    )
    parser.add_argument(
        "--sadece-on-isleme",
        action="store_true",
        help="Random Forest modelini çalıştırmaz",
    )
    parser.add_argument(
        "--uzerine-yaz",
        action="store_true",
        help="Mevcut özellik dosyasını yeniden oluşturur",
    )
    return parser.parse_args()


def komutu_calistir(komut):
    subprocess.run(komut, cwd=proje_klasoru, check=True)


def main():
    ayarlar = ayarlari_oku()

    if ayarlar.hizli_test:
        cikti = proje_klasoru / "output" / "phase3" / "smoke_features.csv.gz"
        model_ciktisi = proje_klasoru / "output" / "phase3" / "rf_smoke"
    else:
        cikti = (
            proje_klasoru
            / "output"
            / "phase3"
            / "breizh_pilot_features.csv.gz"
        )
        model_ciktisi = proje_klasoru / "output" / "phase3" / "rf_pilot"

    on_isleme_komutu = [
        sys.executable,
        str(proje_klasoru / "analysis" / "phase3_breizh_preprocessing.py"),
        "--veri-klasoru",
        str(ayarlar.veri_klasoru),
        "--oran",
        str(ayarlar.oran),
        "--cikti",
        str(cikti),
    ]

    if ayarlar.hizli_test:
        on_isleme_komutu.extend(["--sinif-basina-en-fazla", "2"])
    if ayarlar.uzerine_yaz:
        on_isleme_komutu.append("--uzerine-yaz")

    print("BreizhCrops ön işleme çalıştırılıyor...")
    komutu_calistir(on_isleme_komutu)

    if not ayarlar.sadece_on_isleme:
        print("\nRandom Forest modeli çalıştırılıyor...")
        komutu_calistir(
            [
                sys.executable,
                str(proje_klasoru / "analysis" / "phase3_breizh_rf.py"),
                "--girdi",
                str(cikti),
                "--cikti-klasoru",
                str(model_ciktisi),
            ]
        )

    print("\nBreizhCrops pilot çalışması tamamlandı.")


if __name__ == "__main__":
    main()
