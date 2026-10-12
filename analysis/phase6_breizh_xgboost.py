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
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# ============================================================
# 1. AYARLAR
# ============================================================


def ayarlari_oku():
    parser = argparse.ArgumentParser(
        description="Tam BreizhCrops özellikleriyle XGBoost modeli eğitir."
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
        default=proje_klasoru / "output" / "phase6" / "xgboost_full",
    )
    parser.add_argument("--agac-sayisi", type=int, default=600)
    parser.add_argument("--maksimum-derinlik", type=int, default=8)
    parser.add_argument("--ogrenme-orani", type=float, default=0.05)
    parser.add_argument("--erken-durdurma", type=int, default=30)
    parser.add_argument(
        "--final-testi-calistir",
        action="store_true",
        help="Yalnız bütün model seçimleri bittikten sonra FRH04'ü değerlendirir",
    )
    return parser.parse_args()


# ============================================================
# 2. SINIFLAR VE DEĞERLENDİRME
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
    gercek_matris = np.eye(len(sinif_numaralari), dtype=np.float32)[y_gercek]
    return float(np.mean(np.sum((olasiliklar - gercek_matris) ** 2, axis=1)))


def ece_hesapla(y_gercek, tahmin, olasiliklar, kutu_sayisi=10):
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
            ece += float(maske.mean()) * abs(
                float(dogru_mu[maske].mean()) - float(guven[maske].mean())
            )

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

    print("\n--- XGBOOST -", veri_adi.upper(), "---")
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
    ).plot(ax=ax, cmap="Greens", xticks_rotation=40, colorbar=False)
    plt.title("XGBoost - " + veri_adi)
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
# 3. VERİYİ HAZIRLAMA
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

    # Medyanlar sadece eğitim verisinden öğrenilir.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train).astype(np.float32, copy=False)
    X_valid = imputer.transform(X_valid).astype(np.float32, copy=False)
    if X_test is not None:
        X_test = imputer.transform(X_test).astype(np.float32, copy=False)

    print("Train eksik değer:", int(np.isnan(X_train).sum()))
    print("Valid eksik değer:", int(np.isnan(X_valid).sum()))

    # Az görülen sınıfların eğitimde tamamen kaybolmaması için satır ağırlığı.
    satir_agirliklari = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    ).astype(np.float32)

    # ========================================================
    # 4. XGBOOST MODELİ
    # ========================================================

    print("\nXGBoost modeli eğitiliyor...")
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(sinif_numaralari),
        n_estimators=ayarlar.agac_sayisi,
        max_depth=ayarlar.maksimum_derinlik,
        learning_rate=ayarlar.ogrenme_orani,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=2,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=ayarlar.erken_durdurma,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=satir_agirliklari,
        eval_set=[(X_valid, y_valid)],
        verbose=25,
    )

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
    # 5. ÖZELLİK ÖNEMİ VE SONUÇLAR
    # ========================================================

    onem = pd.DataFrame(
        {"ozellik": kullanilabilir, "onem": model.feature_importances_}
    ).sort_values("onem", ascending=False)
    onem.to_csv(
        ayarlar.cikti_klasoru / "feature_importance.csv",
        index=False,
    )

    ilk_20 = onem.head(20).sort_values("onem")
    plt.figure(figsize=(10, 7))
    plt.barh(ilk_20["ozellik"], ilk_20["onem"], color="forestgreen")
    plt.title("BreizhCrops XGBoost - En Önemli 20 Özellik")
    plt.xlabel("Özellik önemi")
    plt.tight_layout()
    plt.savefig(
        ayarlar.cikti_klasoru / "feature_importance.png",
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
        "n_estimators": ayarlar.agac_sayisi,
        "max_depth": ayarlar.maksimum_derinlik,
        "learning_rate": ayarlar.ogrenme_orani,
        "early_stopping_rounds": ayarlar.erken_durdurma,
        "best_iteration": int(model.best_iteration),
        "class_weight": "balanced sample weights",
        "random_state": 42,
    }
    sonuclar["siniflar"] = sinif_tablosu

    sonuc_dosyasi = ayarlar.cikti_klasoru / "xgboost_results.json"
    sonuc_dosyasi.write_text(
        json.dumps(sonuclar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nSonuçlar kaydedildi:", ayarlar.cikti_klasoru)


if __name__ == "__main__":
    main()
