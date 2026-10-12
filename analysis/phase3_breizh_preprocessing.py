import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ============================================================
# 1. SABİT BİLGİLER
# ============================================================

proje_klasoru = Path(__file__).resolve().parents[1]
indeks_klasoru = proje_klasoru / "data" / "phase1" / "breizhcrops"

# HDF5 dosyasındaki ilk 13 sütunun sırası BreizhCrops belgesinden alınmıştır.
bant_isimleri = [
    "B1", "B10", "B11", "B12", "B2", "B3", "B4",
    "B5", "B6", "B7", "B8", "B8A", "B9",
]

indeks_isimleri = ["NDVI", "NDRE", "NDMI"]
kanal_isimleri = bant_isimleri + indeks_isimleri
bolgeler = ["frh01", "frh02", "frh03", "frh04"]


# ============================================================
# 2. KOMUT SATIRI AYARLARI
# ============================================================


def ayarlari_oku():
    parser = argparse.ArgumentParser(
        description="BreizhCrops pilot verisinden fenolojik özellik çıkarır."
    )
    parser.add_argument(
        "--veri-klasoru",
        type=Path,
        default=Path(r"D:\Datasets\BreizhCrops\2017\L1C"),
        help="frh01.h5-frh04.h5 dosyalarının bulunduğu klasör",
    )
    parser.add_argument(
        "--oran",
        type=float,
        default=0.10,
        help="Her bölge ve sınıftan alınacak örnek oranı",
    )
    parser.add_argument(
        "--tohum",
        type=int,
        default=42,
        help="Tekrarlanabilir örnekleme için rastgelelik değeri",
    )
    parser.add_argument(
        "--sinif-basina-en-az",
        type=int,
        default=20,
        help="Azınlık sınıflarını korumak için bölge-sınıf başına alt sınır",
    )
    parser.add_argument(
        "--sinif-basina-en-fazla",
        type=int,
        default=None,
        help="Hızlı denemede her bölge-sınıftan alınacak üst sınır",
    )
    parser.add_argument(
        "--cikti",
        type=Path,
        default=proje_klasoru / "output" / "phase3" / "breizh_pilot_features.csv.gz",
        help="Çıkarılan özelliklerin kaydedileceği CSV.GZ dosyası",
    )
    parser.add_argument(
        "--uzerine-yaz",
        action="store_true",
        help="Aynı isimde çıktı varsa yeniden oluştur",
    )
    return parser.parse_args()


# ============================================================
# 3. SINIF EŞLEMESİ VE PİLOT ÖRNEKLEM
# ============================================================


def sinif_eslemesini_oku():
    esleme = pd.read_csv(indeks_klasoru / "classmapping.csv")
    esleme = esleme[["id", "classname", "code"]].copy()
    esleme = esleme.rename(
        columns={"id": "class_id", "classname": "class_name"}
    )
    return esleme


def pilot_ornekle(
    oran,
    tohum,
    sinif_basina_en_az=20,
    sinif_basina_en_fazla=None,
):
    if oran <= 0 or oran > 1:
        raise ValueError("--oran değeri 0'dan büyük ve 1'den küçük/eşit olmalıdır.")

    esleme = sinif_eslemesini_oku()
    secilen_bolgeler = []

    for bolge_no, bolge in enumerate(bolgeler):
        indeks = pd.read_csv(indeks_klasoru / (bolge + ".csv"))
        indeks = indeks.merge(
            esleme,
            left_on="CODE_CULTU",
            right_on="code",
            how="inner",
        )
        indeks["region"] = bolge.upper()

        secilen_siniflar = []
        for class_id, grup in indeks.groupby("class_id", sort=True):
            adet = max(1, int(round(len(grup) * oran)))
            adet = max(adet, sinif_basina_en_az)
            if sinif_basina_en_fazla is not None:
                adet = min(adet, sinif_basina_en_fazla)
            adet = min(adet, len(grup))

            # Her çalıştırmada aynı parseller seçilsin diye random_state sabittir.
            grup_tohumu = tohum + (bolge_no * 100) + int(class_id)
            secilen = grup.sample(n=adet, random_state=grup_tohumu)
            secilen_siniflar.append(secilen)

        secilen_bolge = pd.concat(secilen_siniflar, ignore_index=True)
        secilen_bolgeler.append(secilen_bolge)

        print(
            bolge.upper(),
            "eşlenen parsel:", len(indeks),
            "| seçilen pilot:", len(secilen_bolge),
        )

    pilot = pd.concat(secilen_bolgeler, ignore_index=True)
    return pilot


# ============================================================
# 4. BULUT KONTROLÜ VE SPEKTRAL İNDEKSLER
# ============================================================


def guvenli_oran(pay, payda):
    sonuc = np.full(pay.shape, np.nan, dtype=np.float32)
    uygun = np.abs(payda) > 1e-8
    sonuc[uygun] = pay[uygun] / payda[uygun]
    return sonuc


def temiz_gozlem_maskesi(qa60):
    # QA60 değerinin bit kontrolü yalnız tam sayı değerlerde güvenilirdir.
    qa_yuvarlak = np.rint(qa60)
    tam_sayi = np.isfinite(qa60) & np.isclose(qa60, qa_yuvarlak)
    qa_tam = np.where(tam_sayi, qa_yuvarlak, 0).astype(np.int64)

    bulut = (qa_tam & 1024) != 0
    ince_bulut = (qa_tam & 2048) != 0
    temiz = tam_sayi & ~bulut & ~ince_bulut
    return temiz


def kanallari_hazirla(dizi):
    # Bantları 0-1 civarına getiriyoruz; aykırı değerleri burada kesmiyoruz.
    bantlar = dizi[:, :13].astype(np.float32) / 10000.0

    b11 = bantlar[:, 2]
    b2 = bantlar[:, 4]
    b3 = bantlar[:, 5]
    b4 = bantlar[:, 6]
    b5 = bantlar[:, 7]
    b8 = bantlar[:, 10]
    b8a = bantlar[:, 11]

    ndvi = guvenli_oran(b8 - b4, b8 + b4)
    ndre = guvenli_oran(b8a - b5, b8a + b5)
    ndmi = guvenli_oran(b8 - b11, b8 + b11)

    # B2 ve B3 bu aşamada doğrudan indeks hesabında kullanılmıyor.
    # İsimleri yukarıdaki bant sırasını açık göstermek için tutuldu.
    _ = b2, b3

    return np.column_stack([bantlar, ndvi, ndre, ndmi]).astype(np.float32)


# ============================================================
# 5. 24 DÖNEM VE FENOLOJİK ÖZELLİKLER
# ============================================================


def donem_numaralari(tarihler):
    # Her ayı 1-15 ve 16-ay sonu olarak iki parçaya ayırıyoruz.
    sonuc = np.full(len(tarihler), -1, dtype=int)
    uygun = ~pd.isna(tarihler)
    ay = tarihler[uygun].month.to_numpy()
    gun = tarihler[uygun].day.to_numpy()
    sonuc[uygun] = ((ay - 1) * 2 + (gun > 15).astype(int)).astype(int)
    return sonuc


def egim_hesapla(x, y):
    uygun = np.isfinite(y)
    if uygun.sum() < 2:
        return np.nan
    return float(np.polyfit(x[uygun], y[uygun], 1)[0])


def alan_hesapla(x, y):
    # Trapez yöntemi: komşu iki noktanın ortalaması × aralarındaki uzaklık.
    if len(y) < 2:
        return 0.0
    return float(np.sum(((y[1:] + y[:-1]) / 2) * np.diff(x)))


def fenolojik_ozellikler(seri, on_ek):
    sonuc = {}
    uygun = np.isfinite(seri)

    if uygun.sum() == 0:
        isimler = [
            "ortalama", "medyan", "standart_sapma", "minimum", "maksimum",
            "genlik", "yuzde25", "yuzde75", "alan", "tepe_donemi",
            "yesillenme_egimi", "yaslanma_egimi",
        ]
        for isim in isimler:
            sonuc[on_ek + "_" + isim] = np.nan
        return sonuc

    x = np.arange(24, dtype=np.float32)
    x_uygun = x[uygun]
    y_uygun = seri[uygun]
    tepe_donemi = int(x_uygun[np.argmax(y_uygun)])

    sonuc[on_ek + "_ortalama"] = float(np.mean(y_uygun))
    sonuc[on_ek + "_medyan"] = float(np.median(y_uygun))
    sonuc[on_ek + "_standart_sapma"] = float(np.std(y_uygun))
    sonuc[on_ek + "_minimum"] = float(np.min(y_uygun))
    sonuc[on_ek + "_maksimum"] = float(np.max(y_uygun))
    sonuc[on_ek + "_genlik"] = float(np.max(y_uygun) - np.min(y_uygun))
    sonuc[on_ek + "_yuzde25"] = float(np.percentile(y_uygun, 25))
    sonuc[on_ek + "_yuzde75"] = float(np.percentile(y_uygun, 75))
    sonuc[on_ek + "_alan"] = alan_hesapla(x_uygun, y_uygun)
    sonuc[on_ek + "_tepe_donemi"] = tepe_donemi + 1

    once = uygun & (x <= tepe_donemi)
    sonra = uygun & (x >= tepe_donemi)
    sonuc[on_ek + "_yesillenme_egimi"] = egim_hesapla(x[once], seri[once])
    sonuc[on_ek + "_yaslanma_egimi"] = egim_hesapla(x[sonra], seri[sonra])
    return sonuc


def parsel_ozelliklerini_cikar(dizi, satir):
    if dizi.ndim != 2 or dizi.shape[1] < 17 or len(dizi) == 0:
        return None

    tarihler = pd.to_datetime(dizi[:, 16], unit="ns", errors="coerce")
    tarih_var = ~pd.isna(tarihler)
    temiz = temiz_gozlem_maskesi(dizi[:, 15]) & tarih_var

    if temiz.sum() == 0:
        return None

    kanallar = kanallari_hazirla(dizi)
    donemler = donem_numaralari(tarihler)

    sonuc = {
        "parcel_id": int(satir["id"]),
        "region": satir["region"],
        "crop_code": satir["CODE_CULTU"],
        "class_id": int(satir["class_id"]),
        "class_name": satir["class_name"],
        "sequence_length": int(len(dizi)),
        "valid_observation_count": int(temiz.sum()),
        "invalid_observation_rate": float(1 - (temiz.sum() / len(dizi))),
    }

    donem_medyani = np.full((24, len(kanal_isimleri)), np.nan, dtype=np.float32)

    for donem in range(24):
        bu_donem = donemler == donem
        temiz_bu_donem = bu_donem & temiz

        sonuc[f"donem_{donem + 1:02d}_toplam_gozlem"] = int(bu_donem.sum())
        sonuc[f"donem_{donem + 1:02d}_gecerli_gozlem"] = int(
            temiz_bu_donem.sum()
        )

        if bu_donem.sum() == 0:
            bulut_orani = np.nan
        else:
            bulut_orani = 1 - (temiz_bu_donem.sum() / bu_donem.sum())
        sonuc[f"donem_{donem + 1:02d}_gecersiz_oran"] = bulut_orani

        if temiz_bu_donem.sum() > 0:
            donem_medyani[donem] = np.nanmedian(
                kanallar[temiz_bu_donem], axis=0
            )

        for kanal_no, kanal in enumerate(kanal_isimleri):
            sonuc[f"donem_{donem + 1:02d}_{kanal}"] = float(
                donem_medyani[donem, kanal_no]
            )

    # Fenolojik özetleri bitki gelişimini en iyi gösteren indekslerden çıkarıyoruz.
    for indeks in indeks_isimleri:
        kanal_no = kanal_isimleri.index(indeks)
        seri = donem_medyani[:, kanal_no]
        sonuc.update(fenolojik_ozellikler(seri, indeks))

    return sonuc


# ============================================================
# 6. HDF5 DOSYALARINI PARÇA PARÇA OKUMA
# ============================================================


def ozellikleri_kaydet(pilot, veri_klasoru, cikti_dosyasi):
    cikti_dosyasi.parent.mkdir(parents=True, exist_ok=True)

    ilk_yazim = True
    tampon = []
    yazilan = 0
    atlanan = 0

    for bolge in bolgeler:
        bolge_pilot = pilot[pilot["region"] == bolge.upper()].copy()
        h5_dosyasi = veri_klasoru / (bolge + ".h5")

        print("\nOkunan dosya:", h5_dosyasi)
        with h5py.File(h5_dosyasi, "r") as h5:
            for _, satir in bolge_pilot.iterrows():
                veri_yolu = str(satir["path"])
                if veri_yolu not in h5:
                    atlanan += 1
                    continue

                dizi = h5[veri_yolu][:]
                ozellikler = parsel_ozelliklerini_cikar(dizi, satir)

                if ozellikler is None:
                    atlanan += 1
                    continue

                tampon.append(ozellikler)

                # Tüm pilotu belleğe almamak için 250 satırda bir diske yazıyoruz.
                if len(tampon) == 250:
                    pd.DataFrame(tampon).to_csv(
                        cikti_dosyasi,
                        index=False,
                        compression="gzip",
                        mode="w" if ilk_yazim else "a",
                        header=ilk_yazim,
                    )
                    yazilan += len(tampon)
                    tampon = []
                    ilk_yazim = False
                    print("Kaydedilen parsel:", yazilan)

    if len(tampon) > 0:
        pd.DataFrame(tampon).to_csv(
            cikti_dosyasi,
            index=False,
            compression="gzip",
            mode="w" if ilk_yazim else "a",
            header=ilk_yazim,
        )
        yazilan += len(tampon)

    print("\nToplam kaydedilen parsel:", yazilan)
    print("Atlanan parsel:", atlanan)
    return yazilan, atlanan


# ============================================================
# 7. PROGRAMI ÇALIŞTIRMA
# ============================================================


def main():
    ayarlar = ayarlari_oku()

    gerekli_dosyalar = [ayarlar.veri_klasoru / (b + ".h5") for b in bolgeler]
    eksik_dosyalar = [dosya for dosya in gerekli_dosyalar if not dosya.is_file()]
    if len(eksik_dosyalar) > 0:
        print("Eksik HDF5 dosyaları:")
        for dosya in eksik_dosyalar:
            print(" -", dosya)
        raise SystemExit(1)

    if ayarlar.cikti.exists() and not ayarlar.uzerine_yaz:
        raise FileExistsError(
            f"Çıktı zaten var: {ayarlar.cikti}\n"
            "Yeniden oluşturmak için --uzerine-yaz kullanın."
        )

    print("BreizhCrops pilot ön işleme başlıyor...")
    print("Pilot oranı:", ayarlar.oran)
    print("Rastgelelik tohumu:", ayarlar.tohum)

    pilot = pilot_ornekle(
        ayarlar.oran,
        ayarlar.tohum,
        ayarlar.sinif_basina_en_az,
        ayarlar.sinif_basina_en_fazla,
    )

    print("\nPilot sınıf ve bölge dağılımı:")
    print(pd.crosstab(pilot["class_name"], pilot["region"]))

    ayarlar.cikti.parent.mkdir(parents=True, exist_ok=True)
    manifest_dosyasi = ayarlar.cikti.with_name(
        ayarlar.cikti.name.replace(".csv.gz", "_manifest.csv")
    )
    pilot[[
        "id", "region", "CODE_CULTU", "class_id", "class_name", "path"
    ]].to_csv(manifest_dosyasi, index=False)

    yazilan, atlanan = ozellikleri_kaydet(
        pilot,
        ayarlar.veri_klasoru,
        ayarlar.cikti,
    )

    ozet = {
        "pilot_orani": ayarlar.oran,
        "sinif_basina_en_az": ayarlar.sinif_basina_en_az,
        "rastgelelik_tohumu": ayarlar.tohum,
        "secilen_parsel": int(len(pilot)),
        "kaydedilen_parsel": int(yazilan),
        "atlanan_parsel": int(atlanan),
        "bolge_ayrimi": {
            "egitim": ["FRH01", "FRH02"],
            "dogrulama": ["FRH03"],
            "test": ["FRH04"],
        },
        "sinif_sayisi": 9,
        "donem_sayisi": 24,
        "kanallar": kanal_isimleri,
    }
    ozet_dosyasi = ayarlar.cikti.with_name(
        ayarlar.cikti.name.replace(".csv.gz", "_summary.json")
    )
    ozet_dosyasi.write_text(
        json.dumps(ozet, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Özellik dosyası:", ayarlar.cikti)
    print("Manifest dosyası:", manifest_dosyasi)
    print("Özet dosyası:", ozet_dosyasi)


if __name__ == "__main__":
    main()
