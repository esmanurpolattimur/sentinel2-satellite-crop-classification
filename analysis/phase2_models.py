import json
import os
from pathlib import Path

proje_klasoru = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(proje_klasoru / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. DOSYA YOLLARI
# ============================================================

veri_klasoru = proje_klasoru / "data" / "phase1" / "tum_croptypes"
cikti_klasoru = proje_klasoru / "output" / "phase2"
cikti_klasoru.mkdir(parents=True, exist_ok=True)

egitim_dosyasi = veri_klasoru / "data2016-2018.xlsx"
test_dosyasi = veri_klasoru / "TestData.xlsx"


# ============================================================
# 2. VERİ SETİNİ OKUMA
# ============================================================

print("TUM CropTypes verileri okunuyor...")

train_df = pd.read_excel(egitim_dosyasi)
test_df = pd.read_excel(test_dosyasi)

print("Eğitim/geliştirme boyutu:", train_df.shape)
print("Bağımsız test boyutu:", test_df.shape)
print("Eğitim tekrar eden satır:", train_df.duplicated().sum())
print("Test tekrar eden satır:", test_df.duplicated().sum())

train_df = train_df.drop_duplicates().copy()
test_df = test_df.drop_duplicates().copy()


# ============================================================
# 3. SINIFLARI DÜZENLEME
# ============================================================

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

siniflar = {
    600: "patates",
    131: "kışlık arpa",
    400: "mısır",
    311: "kışlık kolza",
    115: "kışlık buğday",
    603: "şeker pancarı",
    1: "diğer",
}

sinif_isimleri = list(siniflar.values())
sinif_numaralari = {
    sinif_adi: numara for numara, sinif_adi in enumerate(sinif_isimleri)
}


def siniflari_hazirla(df):
    df = df.copy()
    df["mapped_code"] = df["NC"].astype(int).replace(kod_degistirme)

    ana_kodlar = [600, 131, 400, 311, 115, 603]
    df.loc[~df["mapped_code"].isin(ana_kodlar), "mapped_code"] = 1

    df["mapped_class"] = df["mapped_code"].map(siniflar)
    df["target"] = df["mapped_class"].map(sinif_numaralari)
    return df


train_df = siniflari_hazirla(train_df)
test_df = siniflari_hazirla(test_df)

print("\nEğitim sınıf dağılımı:")
print(train_df["mapped_class"].value_counts())

print("\nBağımsız test sınıf dağılımı:")
print(test_df["mapped_class"].value_counts())


# ============================================================
# 4. X VE y OLUŞTURMA
# ============================================================

# Modelde yalnız sayısal zaman serisi ortalamalarını kullanıyoruz.
ozellik_kolonlari = [
    kolon
    for kolon in train_df.columns
    if str(kolon).endswith("_mean") and kolon in test_df.columns
]

X = train_df[ozellik_kolonlari].copy()
y = train_df["target"].copy()

X_bagimsiz_test = test_df[ozellik_kolonlari].copy()
y_bagimsiz_test = test_df["target"].copy()

print("\nÖzellik sayısı:", len(ozellik_kolonlari))
print("X boyutu:", X.shape)
print("y boyutu:", y.shape)


# ============================================================
# 5. EĞİTİM VE DOĞRULAMA OLARAK AYIRMA
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nX_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("X_test :", X_bagimsiz_test.shape)


# ============================================================
# 6. EKSİK DEĞERLERİ DOLDURMA
# ============================================================

imputer = SimpleImputer(strategy="median")

X_train_dolu = imputer.fit_transform(X_train)
X_valid_dolu = imputer.transform(X_valid)
X_test_dolu = imputer.transform(X_bagimsiz_test)

print("\nTrain eksik değer:", np.isnan(X_train_dolu).sum())
print("Valid eksik değer:", np.isnan(X_valid_dolu).sum())
print("Test eksik değer :", np.isnan(X_test_dolu).sum())


# ============================================================
# 7. MODEL SONUÇLARINI GÖSTEREN FONKSİYON
# ============================================================

tum_sonuclar = {}


def modeli_degerlendir(model, X_deger, y_deger, model_adi, veri_adi):
    tahmin = model.predict(X_deger)

    accuracy = accuracy_score(y_deger, tahmin)
    macro_f1 = f1_score(
        y_deger,
        tahmin,
        labels=list(range(len(sinif_isimleri))),
        average="macro",
        zero_division=0,
    )
    balanced_accuracy = balanced_accuracy_score(y_deger, tahmin)

    print("\n---", model_adi, "-", veri_adi, "---")
    print("Accuracy:", round(accuracy, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("Balanced Accuracy:", round(balanced_accuracy, 4))
    print("\nClassification Report:")
    print(
        classification_report(
            y_deger,
            tahmin,
            labels=list(range(len(sinif_isimleri))),
            target_names=sinif_isimleri,
            zero_division=0,
        )
    )

    matris = confusion_matrix(
        y_deger,
        tahmin,
        labels=list(range(len(sinif_isimleri))),
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(
        confusion_matrix=matris,
        display_labels=sinif_isimleri,
    ).plot(ax=ax, cmap="Blues", xticks_rotation=35, colorbar=False)
    plt.title(model_adi + " - " + veri_adi)
    plt.tight_layout()

    dosya_adi = (
        model_adi.lower().replace(" ", "_")
        + "_"
        + veri_adi.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )
    plt.savefig(cikti_klasoru / dosya_adi, dpi=160, bbox_inches="tight")
    plt.close()

    anahtar = model_adi + " - " + veri_adi
    tum_sonuclar[anahtar] = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "confusion_matrix": matris.tolist(),
    }


# ============================================================
# 8. RANDOM FOREST MODELİ
# ============================================================

print("\nRandom Forest modeli eğitiliyor...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=1,
)

rf_model.fit(X_train_dolu, y_train)

modeli_degerlendir(
    rf_model, X_valid_dolu, y_valid, "Random Forest", "Doğrulama"
)
modeli_degerlendir(
    rf_model, X_test_dolu, y_bagimsiz_test, "Random Forest", "Bağımsız Test"
)


# Random Forest modelinin en önemli 20 özelliği.
ozellik_onemi = pd.DataFrame(
    {
        "ozellik": ozellik_kolonlari,
        "onem": rf_model.feature_importances_,
    }
).sort_values("onem", ascending=False)

ozellik_onemi.to_csv(
    cikti_klasoru / "random_forest_ozellik_onemi.csv",
    index=False,
    encoding="utf-8-sig",
)

ilk_20 = ozellik_onemi.head(20).sort_values("onem")
plt.figure(figsize=(9, 7))
plt.barh(ilk_20["ozellik"], ilk_20["onem"], color="#375a7f")
plt.title("Random Forest En Önemli 20 Özellik")
plt.xlabel("Özellik önemi")
plt.tight_layout()
plt.savefig(
    cikti_klasoru / "random_forest_ozellik_onemi.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 9. MLP MODELİ
# ============================================================

# MLP için sayısal değerleri standartlaştırıyoruz.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_dolu)
X_valid_scaled = scaler.transform(X_valid_dolu)
X_test_scaled = scaler.transform(X_test_dolu)

print("\nMLP modeli eğitiliyor...")

mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    max_iter=250,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42,
)

mlp_model.fit(X_train_scaled, y_train)

modeli_degerlendir(
    mlp_model, X_valid_scaled, y_valid, "MLP", "Doğrulama"
)
modeli_degerlendir(
    mlp_model, X_test_scaled, y_bagimsiz_test, "MLP", "Bağımsız Test"
)


# ============================================================
# 10. SONUÇLARI KAYDETME
# ============================================================

sonuc_dosyasi = cikti_klasoru / "tum_baseline_results.json"

with open(sonuc_dosyasi, "w", encoding="utf-8") as dosya:
    json.dump(tum_sonuclar, dosya, indent=2, ensure_ascii=False)

print("\nModeller tamamlandı.")
print("Sonuç dosyası:", sonuc_dosyasi)
