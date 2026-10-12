import argparse
import copy
import gc
import json
import os
from pathlib import Path

proje_klasoru = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(proje_klasoru / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. AYARLAR VE SABİTLER
# ============================================================


def ayarlari_oku():
    parser = argparse.ArgumentParser(
        description="BreizhCrops 24 dönemlik zaman serisiyle 1D-CNN eğitir."
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
        default=proje_klasoru / "output" / "phase7" / "cnn_full",
    )
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch-boyutu", type=int, default=512)
    parser.add_argument("--ogrenme-orani", type=float, default=0.001)
    parser.add_argument("--sabir", type=int, default=5)
    parser.add_argument(
        "--final-testi-calistir",
        action="store_true",
        help="Yalnız bütün model seçimleri bittikten sonra FRH04'ü değerlendirir",
    )
    return parser.parse_args()


kanallar = [
    "B1",
    "B10",
    "B11",
    "B12",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "NDVI",
    "NDRE",
    "NDMI",
]

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


def zaman_kolonlarini_olustur():
    """24 dönemi ve 16 kanalı doğru zaman sırasıyla listeler."""
    return [
        f"donem_{donem:02d}_{kanal}"
        for donem in range(1, 25)
        for kanal in kanallar
    ]


# ============================================================
# 2. 1D-CNN MİMARİSİ
# ============================================================


class BirBoyutluCNN(nn.Module):
    def __init__(self, kanal_sayisi, sinif_sayisi):
        super().__init__()
        self.ozellik_cikarici = nn.Sequential(
            nn.Conv1d(kanal_sayisi, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.siniflandirici = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(64, sinif_sayisi),
        )

    def forward(self, x):
        x = self.ozellik_cikarici(x)
        return self.siniflandirici(x)


# ============================================================
# 3. YARDIMCI FONKSİYONLAR
# ============================================================


def tensore_cevir(X):
    # CSV sırası: örnek, dönem, kanal. Conv1d sırası: örnek, kanal, dönem.
    X = X.reshape(-1, 24, len(kanallar)).transpose(0, 2, 1)
    return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))


def tahmin_yap(model, veri_yukleyici, cihaz):
    model.eval()
    gercekler = []
    olasiliklar = []

    with torch.no_grad():
        for X_batch, y_batch in veri_yukleyici:
            X_batch = X_batch.to(cihaz)
            logits = model(X_batch)
            batch_olasilik = torch.softmax(logits, dim=1)
            olasiliklar.append(batch_olasilik.cpu().numpy())
            gercekler.append(y_batch.numpy())

    return np.concatenate(gercekler), np.concatenate(olasiliklar)


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


def modeli_degerlendir(model, yukleyici, veri_adi, cikti_klasoru, cihaz):
    y, olasiliklar = tahmin_yap(model, yukleyici, cihaz)
    tahmin = olasiliklar.argmax(axis=1)

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

    print("\n--- 1D-CNN -", veri_adi.upper(), "---")
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
    ).plot(ax=ax, cmap="Oranges", xticks_rotation=40, colorbar=False)
    plt.title("1D-CNN - " + veri_adi)
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
# 4. VERİYİ OKUMA VE DÖNÜŞTÜRME
# ============================================================


def main():
    ayarlar = ayarlari_oku()
    ayarlar.cikti_klasoru.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)

    if not ayarlar.girdi.is_file():
        raise FileNotFoundError("Özellik dosyası bulunamadı: " + str(ayarlar.girdi))

    zaman_kolonlari = zaman_kolonlarini_olustur()
    gerekli_kolonlar = ["region", "class_id"] + zaman_kolonlari

    baslik = pd.read_csv(ayarlar.girdi, nrows=0)
    eksik_kolonlar = [k for k in gerekli_kolonlar if k not in baslik.columns]
    if eksik_kolonlar:
        raise ValueError("Eksik zaman kolonu var: " + str(eksik_kolonlar[:5]))

    veri_tipleri = {kolon: "float32" for kolon in zaman_kolonlari}
    veri_tipleri.update({"region": "category", "class_id": "int16"})

    print("24 dönemlik zaman serileri okunuyor:", ayarlar.girdi)
    df = pd.read_csv(
        ayarlar.girdi,
        usecols=gerekli_kolonlar,
        dtype=veri_tipleri,
    )
    print("Okunan veri boyutu:", df.shape)

    train_maskesi = df["region"].isin(["FRH01", "FRH02"])
    valid_maskesi = df["region"] == "FRH03"
    test_maskesi = df["region"] == "FRH04"

    train_sayisi = int(train_maskesi.sum())
    valid_sayisi = int(valid_maskesi.sum())
    test_sayisi = int(test_maskesi.sum())
    print("Eğitim satırı:", train_sayisi)
    print("Doğrulama satırı:", valid_sayisi)
    print("Ayrılmış final test satırı:", test_sayisi)

    X_train = df.loc[train_maskesi, zaman_kolonlari].to_numpy(
        dtype=np.float32,
        copy=False,
    )
    y_train = df.loc[train_maskesi, "class_id"].to_numpy(dtype=np.int64)
    X_valid = df.loc[valid_maskesi, zaman_kolonlari].to_numpy(
        dtype=np.float32,
        copy=False,
    )
    y_valid = df.loc[valid_maskesi, "class_id"].to_numpy(dtype=np.int64)

    if ayarlar.final_testi_calistir:
        X_test = df.loc[test_maskesi, zaman_kolonlari].to_numpy(
            dtype=np.float32,
            copy=False,
        )
        y_test = df.loc[test_maskesi, "class_id"].to_numpy(dtype=np.int64)
    else:
        X_test = None
        y_test = None

    del df, train_maskesi, valid_maskesi, test_maskesi
    gc.collect()

    # Medyan ve ölçek değerlerinin ikisi de yalnız eğitimden öğrenilir.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train).astype(np.float32, copy=False)
    X_valid = imputer.transform(X_valid).astype(np.float32, copy=False)
    if X_test is not None:
        X_test = imputer.transform(X_test).astype(np.float32, copy=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_valid = scaler.transform(X_valid).astype(np.float32, copy=False)
    if X_test is not None:
        X_test = scaler.transform(X_test).astype(np.float32, copy=False)

    print("Train eksik değer:", int(np.isnan(X_train).sum()))
    print("Valid eksik değer:", int(np.isnan(X_valid).sum()))

    train_verisi = TensorDataset(
        tensore_cevir(X_train),
        torch.from_numpy(y_train),
    )
    valid_verisi = TensorDataset(
        tensore_cevir(X_valid),
        torch.from_numpy(y_valid),
    )
    del X_train, X_valid
    gc.collect()

    train_yukleyici = DataLoader(
        train_verisi,
        batch_size=ayarlar.batch_boyutu,
        shuffle=True,
        num_workers=0,
    )
    valid_yukleyici = DataLoader(
        valid_verisi,
        batch_size=ayarlar.batch_boyutu,
        shuffle=False,
        num_workers=0,
    )

    if X_test is not None:
        test_verisi = TensorDataset(
            tensore_cevir(X_test),
            torch.from_numpy(y_test),
        )
        test_yukleyici = DataLoader(
            test_verisi,
            batch_size=ayarlar.batch_boyutu,
            shuffle=False,
            num_workers=0,
        )
        del X_test
    else:
        test_yukleyici = None

    # ========================================================
    # 5. MODELİ EĞİTME
    # ========================================================

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Kullanılan cihaz:", cihaz)

    sinif_agirliklari = compute_class_weight(
        class_weight="balanced",
        classes=np.array(sinif_numaralari),
        y=y_train,
    ).astype(np.float32)
    agirlik_tensoru = torch.tensor(sinif_agirliklari, device=cihaz)

    model = BirBoyutluCNN(len(kanallar), len(sinif_numaralari)).to(cihaz)
    kayip_fonksiyonu = nn.CrossEntropyLoss(weight=agirlik_tensoru)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ayarlar.ogrenme_orani,
    )

    en_iyi_macro_f1 = -1.0
    en_iyi_agirliklar = None
    iyilesmeyen_epoch = 0
    egitim_kayiplari = []
    valid_macro_f1_degerleri = []

    for epoch in range(1, ayarlar.epoch + 1):
        model.train()
        toplam_kayip = 0.0

        for X_batch, y_batch in train_yukleyici:
            X_batch = X_batch.to(cihaz)
            y_batch = y_batch.to(cihaz)

            optimizer.zero_grad()
            logits = model(X_batch)
            kayip = kayip_fonksiyonu(logits, y_batch)
            kayip.backward()
            optimizer.step()
            toplam_kayip += float(kayip.item()) * len(y_batch)

        epoch_kaybi = toplam_kayip / len(train_verisi)
        valid_y, valid_olasilik = tahmin_yap(model, valid_yukleyici, cihaz)
        valid_tahmin = valid_olasilik.argmax(axis=1)
        valid_macro_f1 = f1_score(
            valid_y,
            valid_tahmin,
            labels=sinif_numaralari,
            average="macro",
            zero_division=0,
        )

        egitim_kayiplari.append(epoch_kaybi)
        valid_macro_f1_degerleri.append(float(valid_macro_f1))
        print(
            f"Epoch {epoch:02d} | kayıp={epoch_kaybi:.4f} | "
            f"valid Macro-F1={valid_macro_f1:.4f}"
        )

        if valid_macro_f1 > en_iyi_macro_f1 + 1e-4:
            en_iyi_macro_f1 = float(valid_macro_f1)
            en_iyi_agirliklar = copy.deepcopy(model.state_dict())
            iyilesmeyen_epoch = 0
        else:
            iyilesmeyen_epoch += 1

        if iyilesmeyen_epoch >= ayarlar.sabir:
            print("Erken durdurma uygulandı.")
            break

    model.load_state_dict(en_iyi_agirliklar)
    torch.save(model.state_dict(), ayarlar.cikti_klasoru / "cnn_model.pt")

    # ========================================================
    # 6. DEĞERLENDİRME VE KAYDETME
    # ========================================================

    sonuclar = {
        "dogrulama": modeli_degerlendir(
            model,
            valid_yukleyici,
            "Doğrulama",
            ayarlar.cikti_klasoru,
            cihaz,
        )
    }

    if ayarlar.final_testi_calistir:
        sonuclar["final_test"] = modeli_degerlendir(
            model,
            test_yukleyici,
            "Final Test",
            ayarlar.cikti_klasoru,
            cihaz,
        )
    else:
        print(
            "\nFRH04 skoru hesaplanmadı. Final model seçilene kadar "
            "--final-testi-calistir kullanılmamalıdır."
        )

    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochlar = range(1, len(egitim_kayiplari) + 1)
    ax1.plot(epochlar, egitim_kayiplari, color="darkorange", label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="darkorange")
    ax2 = ax1.twinx()
    ax2.plot(
        epochlar,
        valid_macro_f1_degerleri,
        color="navy",
        label="Valid Macro-F1",
    )
    ax2.set_ylabel("Valid Macro-F1", color="navy")
    plt.title("BreizhCrops 1D-CNN Eğitim Eğrisi")
    fig.tight_layout()
    plt.savefig(
        ayarlar.cikti_klasoru / "cnn_training_curve.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    sonuclar["veri_boyutlari"] = {
        "egitim": train_sayisi,
        "dogrulama": valid_sayisi,
        "test": test_sayisi,
        "donem_sayisi": 24,
        "kanal_sayisi": len(kanallar),
    }
    sonuclar["model_ayarlari"] = {
        "channels": [len(kanallar), 32, 64],
        "batch_size": ayarlar.batch_boyutu,
        "maximum_epoch": ayarlar.epoch,
        "actual_epoch": len(egitim_kayiplari),
        "learning_rate": ayarlar.ogrenme_orani,
        "early_stopping_patience": ayarlar.sabir,
        "best_validation_macro_f1": en_iyi_macro_f1,
        "device": str(cihaz),
        "random_state": 42,
    }
    sonuclar["siniflar"] = sinif_tablosu

    (ayarlar.cikti_klasoru / "cnn_results.json").write_text(
        json.dumps(sonuclar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nSonuçlar kaydedildi:", ayarlar.cikti_klasoru)


if __name__ == "__main__":
    main()
