import argparse
import subprocess
import sys
from pathlib import Path


# ============================================================
# TAM BREIZHCROPS RANDOM FOREST ÇALIŞTIRMA DOSYASI
# ============================================================

proje_klasoru = Path(__file__).resolve().parent
ozellik_dosyasi = (
    proje_klasoru / "output" / "phase4" / "breizh_full_features.csv.gz"
)
model_ciktisi = proje_klasoru / "output" / "phase4" / "rf_full"


def ayarlari_oku():
    parser = argparse.ArgumentParser(
        description="Tüm BreizhCrops verisini işler ve Random Forest eğitir."
    )
    parser.add_argument(
        "--veri-klasoru",
        type=Path,
        default=Path(r"D:\Datasets\BreizhCrops\2017\L1C"),
    )
    parser.add_argument(
        "--uzerine-yaz",
        action="store_true",
        help="Tam özellik dosyasını baştan oluşturur",
    )
    parser.add_argument(
        "--sadece-on-isleme",
        action="store_true",
        help="Özellikleri çıkarır fakat modeli çalıştırmaz",
    )
    parser.add_argument(
        "--sadece-model",
        action="store_true",
        help="Mevcut tam özellik dosyasını kullanır",
    )
    parser.add_argument(
        "--final-testi-calistir",
        action="store_true",
        help="Model seçimi bittiyse FRH04 final skorunu hesaplar",
    )
    return parser.parse_args()


def komutu_calistir(komut):
    subprocess.run(komut, cwd=proje_klasoru, check=True)


def main():
    ayarlar = ayarlari_oku()

    if ayarlar.sadece_on_isleme and ayarlar.sadece_model:
        raise ValueError(
            "--sadece-on-isleme ve --sadece-model birlikte kullanılamaz."
        )

    if not ayarlar.sadece_model:
        on_isleme_komutu = [
            sys.executable,
            str(proje_klasoru / "analysis" / "phase3_breizh_preprocessing.py"),
            "--veri-klasoru",
            str(ayarlar.veri_klasoru),
            "--oran",
            "1.0",
            "--cikti",
            str(ozellik_dosyasi),
        ]
        if ayarlar.uzerine_yaz:
            on_isleme_komutu.append("--uzerine-yaz")

        print("Tüm BreizhCrops parselleri işleniyor...")
        komutu_calistir(on_isleme_komutu)

    if ayarlar.sadece_on_isleme:
        print("Tam veri ön işleme tamamlandı.")
        return

    if not ozellik_dosyasi.is_file():
        raise FileNotFoundError(
            "Tam özellik dosyası bulunamadı: " + str(ozellik_dosyasi)
        )

    model_komutu = [
        sys.executable,
        str(proje_klasoru / "analysis" / "phase3_breizh_rf.py"),
        "--girdi",
        str(ozellik_dosyasi),
        "--cikti-klasoru",
        str(model_ciktisi),
    ]
    if ayarlar.final_testi_calistir:
        model_komutu.append("--final-testi-calistir")

    print("Tam veri Random Forest modeli çalıştırılıyor...")
    komutu_calistir(model_komutu)
    print("Tam BreizhCrops Random Forest çalışması tamamlandı.")


if __name__ == "__main__":
    main()
