import argparse
import gc
import json
import os
from pathlib import Path

proje_klasoru = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(proje_klasoru / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. AYARLAR
# ============================================================


def ayarlari_oku():
    parser = argparse.ArgumentParser(
        description="Tam BreizhCrops özellikleriyle MLP modeli eğitir."
    )
    parser.add_argument(
        "--girdi",
        type=Path,
        default=(
            proje_klasoru
            / "output"
            / "phase4"
            / "breizh_full_features.csv.gz"
        ),
    )
    parser.add_argument(
        "--cikti-klasoru",
        type=Path,
        default=proje_klasoru / "output" / "phase5" / "mlp_full",
    )
    parser.add_argument("--maksimum-iterasyon", type=int, default=50)
    parser.add_argument("--batch-boyutu", type=int, default=512)
    parser.add_argument(
        "--final-testi-calistir",
        action="store_true",
        help="Yalnız bütün model seçimleri bittikten sonra FRH04'ü değerlendirir",
    )
    return parser.parse_args()


# ============================================================
# 2. SINIFLAR VE DEĞERLENDİRME FONKSİYONLARI
# ============================================================


sinif_tablosu = {
    0: "barley",
    1: "wheat",
    2: "rapeseed",
    3: "corn",
    4: "sunflower",
    5: "orchards",
    6: "nuts",
    7: "permanent meadows",
    8: "temporary meadows",
}
sinif_numaralari = list(sinif_tablosu.keys())
sinif_isimleri = list(sinif_tablosu.values())


def cok_sinifli_brier(y_gercek, olasiliklar):
    """Tahmin olasılıklarının çok sınıflı Brier skorunu hesaplar."""
    gercek_matris = np.eye(len(sinif_numaralari), dtype=np.float32)[y_gercek]
    return float(np.mean(np.sum((olasiliklar - gercek_matris) ** 2, axis=1)))


def ece_hesapla(y_gercek, tahmin, olasiliklar, kutu_sayisi=10):
    """Model güveni ile gerçek doğruluk arasındaki farkı hesaplar."""
    guven = olasiliklar.max(axis=1)
    dogru_mu = (tahmin == y_gercek).astype(np.float32)
    sinirlar = np.linspace(0.0, 1.0, kutu_sayisi + 1)
    ece = 0.0

    for i in range(kutu_sayisi):
        if i == kutu_sayisi - 1:
            maske = (guven >= sinirlar[i]) & (guven <= sinirlar[i + 1])
        else:
            maske = (guven >= sinirlar[i]) & (guven < sinirlar[i + 1])

        if maske.any():
            kutu_agirligi = float(maske.mean())
            kutu_dogrulugu = float(dogru_mu[maske].mean())
            kutu_guveni = float(guven[maske].mean())
            ece += kutu_agirligi * abs(kutu_dogrulugu - kutu_guveni)

    return float(ece)


def modeli_degerlendir(model, X, y, veri_adi, cikti_klasoru):
    tahmin = model.predict(X)
    olasiliklar = model.predict_proba(X)

    accuracy = accuracy_score(y, tahmin)
    macro_f1 = f1_score(
        y,
        tahmin,
        labels=sinif_numaralari,
        average="macro",
        zero_division=0,
    )
    balanced_accuracy = balanced_accuracy_score(y, tahmin)
    nll = log_loss(y, olasiliklar, labels=sinif_numaralari)
    brier = cok_sinifli_brier(y, olasiliklar)
    ece = ece_hesapla(y, tahmin, olasiliklar)

    rapor = classification_report(
        y,
        tahmin,
        labels=sinif_numaralari,
        target_names=sinif_isimleri,
        zero_division=0,
        output_dict=True,
    )

    print("\n--- MLP -", veri_adi.upper(), "---")
    print("Accuracy:", round(accuracy, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("Balanced Accuracy:", round(balanced_accuracy, 4))
    print("NLL:", round(nll, 4))
    print("Brier:", round(brier, 4))
    print("ECE:", round(ece, 4))
    print(
        classification_report(
            y,
            tahmin,
            labels=sinif_numaralari,
            target_names=sinif_isimleri,
            zero_division=0,
        )
    )

    matris = confusion_matrix(y, tahmin, labels=sinif_numaralari)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(
        confusion_matrix=matris,
        display_labels=sinif_isimleri,
    ).plot(ax=ax, cmap="Purples", xticks_rotation=40, colorbar=False)
    plt.title("MLP - " + veri_adi)
    plt.tight_layout()
    plt.savefig(
        cikti_klasoru / (veri_adi.lower() + "_confusion_matrix.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "nll": float(nll),
        "brier": brier,
        "ece": ece,
        "classification_report": rapor,
        "confusion_matrix": matris.tolist(),
    }


# ============================================================
# 3. VERİYİ OKUMA VE BÖLGELERE AYIRMA
# ============================================================


def main():
    ayarlar = ayarlari_oku()
    ayarlar.cikti_klasoru.mkdir(parents=True, exist_ok=True)

    if not ayarlar.girdi.is_file():
        raise FileNotFoundError("Özellik dosyası bulunamadı: " + str(ayarlar.girdi))

    bilgi_kolonlari = [
        "parcel_id",
        "region",
        "crop_code",
        "class_id",
        "class_name",
    ]

    print("Özellik dosyası okunuyor:", ayarlar.girdi)
    baslik = pd.read_csv(ayarlar.girdi, nrows=0)
    sayisal_kolonlar = [
        kolon for kolon in baslik.columns if kolon not in bilgi_kolonlari
    ]
    veri_tipleri = {kolon: "float32" for kolon in sayisal_kolonlar}
    veri_tipleri.update(
        {
            "parcel_id": "int64",
            "region": "category",
            "crop_code": "category",
            "class_id": "int16",
            "class_name": "category",
        }
    )
    df = pd.read_csv(ayarlar.girdi, dtype=veri_tipleri)
    print("Veri boyutu:", df.shape)

    ozellik_kolonlari = [
        kolon for kolon in df.columns if kolon not in bilgi_kolonlari
    ]
    train_maskesi = df["region"].isin(["FRH01", "FRH02"])
    valid_maskesi = df["region"] == "FRH03"
    test_maskesi = df["region"] == "FRH04"

    train_sayisi = int(train_maskesi.sum())
    valid_sayisi = int(valid_maskesi.sum())
    test_sayisi = int(test_maskesi.sum())
    print("Eğitim satırı:", train_sayisi)
    print("Doğrulama satırı:", valid_sayisi)
    print("Ayrılmış final test satırı:", test_sayisi)

    if min(train_sayisi, valid_sayisi, test_sayisi) == 0:
        raise ValueError("Eğitim, doğrulama veya test bölümlerinden biri boş.")

    # Eğitimde tamamen boş olan bir kolonun medyanı hesaplanamaz.
    train_ozellikleri = df.loc[train_maskesi, ozellik_kolonlari]
    kullanilabilir = [
        kolon
        for kolon in ozellik_kolonlari
        if not train_ozellikleri[kolon].isna().all()
    ]
    print("Kullanılan özellik sayısı:", len(kullanilabilir))

    X_train = train_ozellikleri[kullanilabilir].to_numpy(
        dtype=np.float32,
        copy=False,
    )
    y_train = df.loc[train_maskesi, "class_id"].to_numpy(dtype=np.int16)
    del train_ozellikleri

    X_valid = df.loc[valid_maskesi, kullanilabilir].to_numpy(
        dtype=np.float32,
        copy=False,
    )
    y_valid = df.loc[valid_maskesi, "class_id"].to_numpy(dtype=np.int16)

    # FRH04 matrisi ancak kullanıcı açıkça isterse belleğe alınır.
    if ayarlar.final_testi_calistir:
        X_test = df.loc[test_maskesi, kullanilabilir].to_numpy(
            dtype=np.float32,
            copy=False,
        )
        y_test = df.loc[test_maskesi, "class_id"].to_numpy(dtype=np.int16)
    else:
        X_test = None
        y_test = None

    del df, train_maskesi, valid_maskesi, test_maskesi
    gc.collect()

    # ========================================================
    # 4. EKSİK DEĞER DOLDURMA VE STANDARTLAŞTIRMA
    # ========================================================

    # Veri sızıntısını önlemek için iki işlem de yalnız train ile öğrenilir.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train).astype(np.float32, copy=False)
    X_valid = imputer.transform(X_valid).astype(np.float32, copy=False)
    if X_test is not None:
        X_test = imputer.transform(X_test).astype(np.float32, copy=False)

    print("Train eksik değer:", int(np.isnan(X_train).sum()))
    print("Valid eksik değer:", int(np.isnan(X_valid).sum()))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_valid = scaler.transform(X_valid).astype(np.float32, copy=False)
    if X_test is not None:
        X_test = scaler.transform(X_test).astype(np.float32, copy=False)

    # ========================================================
    # 5. MLP MODELİ
    # ========================================================

    print("\nMLP modeli eğitiliyor...")
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        batch_size=ayarlar.batch_boyutu,
        learning_rate="adaptive",
        max_iter=ayarlar.maksimum_iterasyon,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=5,
        random_state=42,
        verbose=True,
    )
    model.fit(X_train, y_train)

    sonuclar = {
        "dogrulama": modeli_degerlendir(
            model,
            X_valid,
            y_valid,
            "Doğrulama",
            ayarlar.cikti_klasoru,
        )
    }

    if ayarlar.final_testi_calistir:
        sonuclar["final_test"] = modeli_degerlendir(
            model,
            X_test,
            y_test,
            "Final Test",
            ayarlar.cikti_klasoru,
        )
    else:
        print(
            "\nFRH04 skoru hesaplanmadı. Final model seçilene kadar "
            "--final-testi-calistir kullanılmamalıdır."
        )

    # ========================================================
    # 6. EĞİTİM EĞRİSİ VE SONUÇ DOSYASI
    # ========================================================

    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_, color="darkorange", label="Eğitim kaybı")
    plt.xlabel("İterasyon")
    plt.ylabel("Log loss")
    plt.title("BreizhCrops MLP Eğitim Eğrisi")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        ayarlar.cikti_klasoru / "mlp_loss_curve.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    sonuclar["veri_boyutlari"] = {
        "egitim": train_sayisi,
        "dogrulama": valid_sayisi,
        "test": test_sayisi,
        "ozellik_sayisi": int(len(kullanilabilir)),
    }
    sonuclar["model_ayarlari"] = {
        "hidden_layer_sizes": [128, 64],
        "batch_size": ayarlar.batch_boyutu,
        "maximum_iteration": ayarlar.maksimum_iterasyon,
        "actual_iteration": int(model.n_iter_),
        "early_stopping": True,
        "random_state": 42,
    }
    sonuclar["siniflar"] = sinif_tablosu

    sonuc_dosyasi = ayarlar.cikti_klasoru / "mlp_results.json"
    sonuc_dosyasi.write_text(
        json.dumps(sonuclar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nSonuçlar kaydedildi:", ayarlar.cikti_klasoru)


if __name__ == "__main__":
    main()
