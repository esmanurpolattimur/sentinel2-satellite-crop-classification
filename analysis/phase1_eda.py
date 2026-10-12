import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. DOSYA YOLLARI
# ============================================================

proje_klasoru = Path(__file__).resolve().parents[1]
veri_klasoru = proje_klasoru / "data" / "phase1"
cikti_klasoru = proje_klasoru / "output" / "phase1"
grafik_klasoru = cikti_klasoru / "assets"

cikti_klasoru.mkdir(parents=True, exist_ok=True)
grafik_klasoru.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. GRAFİK AYARLARI
# ============================================================

def grafik_kaydet(dosya_adi):
    """Grafiği çıktı klasörüne kaydeder."""
    plt.tight_layout()
    plt.savefig(grafik_klasoru / dosya_adi, dpi=180, bbox_inches="tight")
    plt.close()


# ============================================================
# 3. BREIZHCROPS VERİ SETİ
# ============================================================

def breizhcrops_analizi():
    breizh_klasoru = veri_klasoru / "breizhcrops"

    # Ürün kodlarının hangi sınıfa ait olduğunu okuyoruz.
    sinif_tablosu = pd.read_csv(breizh_klasoru / "classmapping.csv", index_col=0)
    kod_sinif = sinif_tablosu.drop_duplicates("code").set_index("code")["classname"]

    bolge_adlari = ["frh01", "frh02", "frh03", "frh04"]
    bolge_verileri = []
    bolge_bilgileri = {}

    for bolge in bolge_adlari:
        df = pd.read_csv(breizh_klasoru / (bolge + ".csv"))
        df["region"] = bolge.upper()
        df["classname"] = df["CODE_CULTU"].map(kod_sinif)
        bolge_verileri.append(df)

        eslesen_df = df[df["classname"].notna()].copy()

        print("\n---", bolge.upper(), "---")
        print("Satır sayısı:", len(df))
        print("Sınıfa eşlenen satır:", len(eslesen_df))
        print("Eksik değer sayısı:", df.isnull().sum().sum())
        print("Tekrar eden id:", df["id"].duplicated().sum())

        bolge_bilgileri[bolge.upper()] = {
            "raw_rows": int(len(df)),
            "mapped_rows": int(len(eslesen_df)),
            "unmapped_rows": int(df["classname"].isna().sum()),
            "duplicate_ids": int(df["id"].duplicated().sum()),
            "missing_by_column": {
                kolon: int(deger) for kolon, deger in df.isna().sum().items()
            },
            "sequence_length": {
                "min": int(df["sequencelength"].min()),
                "median": float(df["sequencelength"].median()),
                "mean": float(df["sequencelength"].mean()),
                "max": int(df["sequencelength"].max()),
            },
        }

    tum_bolgeler = pd.concat(bolge_verileri, ignore_index=True)
    eslesen_veri = tum_bolgeler[tum_bolgeler["classname"].notna()].copy()

    # Sınıfların rapordaki sırası bozulmasın diye classmapping sırasını kullanıyoruz.
    sinif_sirasi = (
        sinif_tablosu.sort_values("id")["classname"].drop_duplicates().tolist()
    )

    sinif_sayilari = pd.crosstab(
        eslesen_veri["classname"], eslesen_veri["region"]
    ).reindex(sinif_sirasi, fill_value=0)
    sinif_sayilari["Toplam"] = sinif_sayilari.sum(axis=1)
    sinif_sayilari.to_csv(
        cikti_klasoru / "breizhcrops_class_counts.csv", encoding="utf-8-sig"
    )

    print("\n--- BREIZHCROPS SINIF DAĞILIMI ---")
    print(sinif_sayilari)

    sinif_sayilari.drop(columns="Toplam").plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        color=["#375a7f", "#4f86a6", "#d7a84b", "#9b5f63"],
    )
    plt.title("BreizhCrops Sınıf Dağılımı")
    plt.xlabel("Ürün sınıfı")
    plt.ylabel("Parsel sayısı")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Bölge")
    grafik_kaydet("breizhcrops_class_distribution.png")

    # Gerçek bir parselin dört bandını çiziyoruz.
    ornek_dosya = proje_klasoru / "breizhcrops_frh04_evidence" / "6082638.csv"
    ornek = pd.read_csv(ornek_dosya)
    ornek["date"] = pd.to_datetime(ornek["doa"])
    ornek = ornek.sort_values("date")

    plt.figure(figsize=(12, 5))
    for bant in ["B2", "B3", "B4", "B8"]:
        plt.plot(ornek["date"], ornek[bant] * 0.0001, label=bant)
    plt.title("BreizhCrops 6082638 Numaralı Parsel")
    plt.xlabel("Tarih")
    plt.ylabel("Yansıtım değeri")
    plt.legend()
    plt.grid(alpha=0.25)
    grafik_kaydet("breizhcrops_real_parcel_6082638.png")

    sonuc = {
        "source_files": [dosya.name for dosya in sorted(breizh_klasoru.glob("*.csv"))],
        "regions": bolge_bilgileri,
        "raw_total_rows": int(len(tum_bolgeler)),
        "mapped_total_rows": int(len(eslesen_veri)),
        "unmapped_total_rows": int(tum_bolgeler["classname"].isna().sum()),
        "class_counts": {
            sinif: int(sayi) for sinif, sayi in sinif_sayilari["Toplam"].items()
        },
        "mapped_missing_by_column": {
            kolon: int(deger)
            for kolon, deger in eslesen_veri.isna().sum().items()
        },
        "sample": {
            "parcel_id": 6082638,
            "rows": int(len(ornek)),
            "columns": ornek.columns.tolist(),
            "start": ornek["date"].min().date().isoformat(),
            "end": ornek["date"].max().date().isoformat(),
            "bands": [kolon for kolon in ornek.columns if kolon.startswith("B")],
            "missing_values": int(ornek.isna().sum().sum()),
            "first_three_records": ornek.drop(columns="date")
            .head(3)
            .to_dict(orient="records"),
        },
    }

    return sonuc


# ============================================================
# 4. TUM CROPTYPES VERİ SETİ
# ============================================================

def tum_croptypes_analizi():
    tum_klasoru = veri_klasoru / "tum_croptypes"

    dosyalar = {
        "Eğitim/geliştirme": tum_klasoru / "data2016-2018.xlsx",
        "Bağımsız test": tum_klasoru / "TestData.xlsx",
    }

    siniflar = {
        600: "patates",
        131: "kışlık arpa",
        400: "mısır",
        311: "kışlık kolza",
        115: "kışlık buğday",
        603: "şeker pancarı",
        1: "diğer",
    }

    # Resmî notebookta aynı sınıfa alınan ham ürün kodları.
    kod_degistirme = {
        601: 600,
        602: 600,
        476: 131,
        411: 400,
        171: 400,
        410: 400,
        177: 400,
        489: 311,
    }

    zamanlar = [
        "feb1", "feb2", "march1", "march2", "april1", "april2", "may1",
        "may2", "june1", "june2", "july1", "july2", "august1", "august2",
    ]
    bantlar = [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
        "B9", "B10", "B11", "B12",
    ]

    sinif_sayilari = {}
    veri_bilgileri = {}
    ornek_bilgisi = None

    for bolum, dosya in dosyalar.items():
        df = pd.read_excel(dosya)

        print("\n--- TUM", bolum.upper(), "---")
        print("Veri seti boyutu:", df.shape)
        print("Eksik hücre:", df.isnull().sum().sum())
        print("Tekrar eden satır:", df.duplicated().sum())

        df["mapped_code"] = df["NC"].astype(int).replace(kod_degistirme)
        ana_kodlar = [600, 131, 400, 311, 115, 603]
        df.loc[~df["mapped_code"].isin(ana_kodlar), "mapped_code"] = 1
        df["mapped_class"] = df["mapped_code"].map(siniflar)

        sinif_sayilari[bolum] = {}
        for sinif in siniflar.values():
            sayi = (df["mapped_class"] == sinif).sum()
            sinif_sayilari[bolum][sinif] = int(sayi)

        spektral_kolonlar = []
        for zaman in zamanlar:
            for bant in bantlar:
                spektral_kolonlar.append(zaman + "_" + bant + "_mean")

        tum_ozellikler = [
            kolon for kolon in df.columns if str(kolon).endswith("_mean")
        ]
        kaynak_veri = df.drop(columns=["mapped_code", "mapped_class"])

        veri_bilgileri[bolum] = {
            "rows": int(len(df)),
            "columns": int(kaynak_veri.shape[1]),
            "raw_label_codes": int(df["NC"].nunique()),
            "mapped_classes": int(df["mapped_class"].nunique()),
            "missing_cells_source": int(kaynak_veri.isna().sum().sum()),
            "duplicate_rows_source": int(kaynak_veri.duplicated().sum()),
            "years": {
                str(yil): int(sayi)
                for yil, sayi in df["Jahr"].value_counts().sort_index().items()
            },
            "feature_columns": int(len(tum_ozellikler)),
            "spectral_only_columns": int(len(spektral_kolonlar)),
            "spectral_tensor_shape": [int(len(df)), len(zamanlar), len(bantlar)],
            "all_values_finite": bool(
                np.isfinite(df[tum_ozellikler].to_numpy(dtype=float)).all()
            ),
        }

        # Eğitim verisinden ortalama değişime yakın bir buğday parseli seçiyoruz.
        if bolum == "Eğitim/geliştirme":
            bugday = df[df["mapped_code"] == 115]
            ndvi_kolonlari = [zaman + "_NDVI_mean" for zaman in zamanlar]
            degisim = bugday[ndvi_kolonlari].std(axis=1)
            ornek_index = (degisim - degisim.median()).abs().idxmin()
            satir = df.loc[ornek_index]

            kayitlar = []
            for zaman in zamanlar:
                kayitlar.append(
                    {
                        "time_bin": zaman,
                        "B2": float(satir[zaman + "_B2_mean"]),
                        "B3": float(satir[zaman + "_B3_mean"]),
                        "B4": float(satir[zaman + "_B4_mean"]),
                        "B8": float(satir[zaman + "_B8_mean"]),
                    }
                )

            ornek_bilgisi = {
                "row_index": int(satir["Unnamed: 0"]),
                "year": int(satir["Jahr"]),
                "raw_code": int(satir["NC"]),
                "raw_label": str(satir["Nutz"]),
                "mapped_class": str(satir["mapped_class"]),
                "selected_records": kayitlar,
            }

            tarihler = pd.date_range(
                str(int(satir["Jahr"])) + "-02-01", periods=len(zamanlar), freq="15D"
            )
            plt.figure(figsize=(12, 5))
            for bant in ["B2", "B3", "B4", "B8"]:
                degerler = [kayit[bant] * 0.0001 for kayit in kayitlar]
                plt.plot(tarihler, degerler, label=bant)
            plt.title("TUM CropTypes Örnek Parsel")
            plt.xlabel("Yaklaşık tarih")
            plt.ylabel("Yansıtım değeri")
            plt.legend()
            plt.grid(alpha=0.25)
            grafik_kaydet("tum_croptypes_real_parcel.png")

    sayim_df = pd.DataFrame(sinif_sayilari).reindex(list(siniflar.values()))
    sayim_df["Toplam"] = sayim_df.sum(axis=1)
    sayim_df.to_csv(
        cikti_klasoru / "tum_croptypes_class_counts.csv", encoding="utf-8-sig"
    )

    print("\n--- TUM SINIF DAĞILIMI ---")
    print(sayim_df)

    sayim_df.drop(columns="Toplam").plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6),
        color=["#375a7f", "#d7a84b"],
    )
    plt.title("TUM CropTypes Sınıf Dağılımı")
    plt.xlabel("Ürün sınıfı")
    plt.ylabel("Parsel sayısı")
    plt.xticks(rotation=30, ha="right")
    grafik_kaydet("tum_croptypes_class_distribution.png")

    sonuc = {
        "official_mapping": {
            str(kod): sinif for kod, sinif in siniflar.items()
        },
        "splits": veri_bilgileri,
        "class_counts": {
            sinif: int(sayi) for sinif, sayi in sayim_df["Toplam"].items()
        },
        "split_class_counts": sinif_sayilari,
        "sample": ornek_bilgisi,
        "critical_limit": (
            "Bağımsız testte şeker pancarı yoktur ve patates yalnız 1 örnektir; "
            "sınıf bazlı test sonucu bu sınıflar için güvenilir değildir."
        ),
    }

    return sonuc


# ============================================================
# 5. İSTEĞE BAĞLI BAVARIANCROPS KONTROLÜ
# ============================================================

def bavariancrops_analizi():
    # Bu veri seti Faz 1 raporuna alınmadı. Klasör varsa yalnız temel sayımları yapıyoruz.
    ana_klasor = veri_klasoru / "bavariancrops" / "holl"
    if not ana_klasor.exists():
        return None

    siniflar = [
        "meadow", "summer barley", "corn", "winter wheat",
        "winter barley", "clover", "winter triticale",
    ]
    bolum_adlari = {"train": "Eğitim", "valid": "Doğrulama", "eval": "Test"}
    bolum_sayilari = {}
    bolum_bilgileri = {}
    ornek_bilgisi = None

    for bolum in ["train", "valid", "eval"]:
        klasor = ana_klasor / bolum
        y = np.load(klasor / "y.npy", allow_pickle=True).reshape(-1)
        ndims = np.load(klasor / "ndims.npy", allow_pickle=True)
        uzunluklar = np.load(
            klasor / "sequencelengths.npy", allow_pickle=True
        ).reshape(-1)
        ids = np.load(klasor / "ids.npy", allow_pickle=True).reshape(-1)
        x = np.load(klasor / "X.npy", allow_pickle=True)

        if y.min() == 1:
            y_duzeltilmis = y.astype(int) - 1
        else:
            y_duzeltilmis = y.astype(int)

        bolum_sayilari[bolum_adlari[bolum]] = {}
        for i, sinif in enumerate(siniflar):
            bolum_sayilari[bolum_adlari[bolum]][sinif] = int(
                (y_duzeltilmis == i).sum()
            )

        bolum_bilgileri[bolum] = {
            "samples": int(len(y)),
            "y_shape": list(y.shape),
            "x_shape": list(x.shape),
            "x_dtype": str(x.dtype),
            "ndims_unique": np.unique(ndims).tolist(),
            "sequence_length": {
                "min": float(np.nanmin(uzunluklar)),
                "median": float(np.nanmedian(uzunluklar)),
                "mean": float(np.nanmean(uzunluklar)),
                "max": float(np.nanmax(uzunluklar)),
            },
            "duplicate_ids": int(pd.Series(ids).duplicated().sum()),
            "missing_y": int(pd.isna(y).sum()),
            "missing_sequence_lengths": int(pd.isna(uzunluklar).sum()),
        }

        if bolum == "train":
            ilk_ornek = np.asarray(x[0], dtype=float)
            if ilk_ornek.ndim == 1:
                ozellik_sayisi = int(np.asarray(ndims)[0])
                ilk_ornek = ilk_ornek.reshape(-1, ozellik_sayisi)

            ornek_bilgisi = {
                "split": bolum,
                "id": str(ids[0]),
                "label_index": int(y_duzeltilmis[0]),
                "label": siniflar[int(y_duzeltilmis[0])],
                "raw_shape": list(np.asarray(x[0]).shape),
                "matrix_shape": list(ilk_ornek.shape),
                "first_three_records": ilk_ornek[:3].tolist(),
                "nan_count": int(np.isnan(ilk_ornek).sum()),
                "inf_count": int(np.isinf(ilk_ornek).sum()),
            }

    sayim_df = pd.DataFrame(bolum_sayilari).reindex(siniflar)
    sayim_df["Toplam"] = sayim_df.sum(axis=1)
    sayim_df.to_csv(
        cikti_klasoru / "bavariancrops_class_counts.csv", encoding="utf-8-sig"
    )

    return {
        "classes": siniflar,
        "splits": bolum_bilgileri,
        "class_counts": {
            sinif: int(sayi) for sinif, sayi in sayim_df["Toplam"].items()
        },
        "total_samples": int(sayim_df["Toplam"].sum()),
        "sample": ornek_bilgisi,
    }


# ============================================================
# 6. ANALİZİ ÇALIŞTIRMA VE SONUÇLARI KAYDETME
# ============================================================

print("Faz 1 veri analizi başlıyor...")

sonuclar = {
    "breizhcrops": breizhcrops_analizi(),
    "tum_croptypes": tum_croptypes_analizi(),
    "bavariancrops": bavariancrops_analizi(),
}

ozet_dosyasi = cikti_klasoru / "eda_summary.json"
with open(ozet_dosyasi, "w", encoding="utf-8") as dosya:
    json.dump(sonuclar, dosya, indent=2, ensure_ascii=False)

print("\nAnaliz tamamlandı.")
print("Özet dosyası:", ozet_dosyasi)
