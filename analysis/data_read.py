import pandas as pd


class_mapping = pd.read_csv(
    "data/phase1/breizhcrops/classmapping.csv"
)

#Csv kaydı sırasında olusan gereksiz indeks sutunu kaldırılır
class_mapping = class_mapping.drop(
    columns=["Unnamed: 0"]
)

print("Eşleme tablosunun boyutu:", class_mapping.shape)
print("Eşleme tablosunun sütunları:", class_mapping.columns.tolist())
print("\nEşleme tablosunun ilk 5 satırı:")
print(class_mapping.head())


frh01 = pd.read_csv(
    "data/phase1/breizhcrops/frh01.csv"
)

print("\nFRH01 tablo boyutu:", frh01.shape)
print("FRH01 sütunları:", frh01.columns.tolist())
print("\nFRH01 ilk 5 satırı:")
print(frh01.head())

# Ham ürün kodları sınıf bilgileriyle eşleştirilir.
#Sol birlestirme secilerek butun satırlar korunur

frh01_mapped = frh01.merge(
    class_mapping,
    how="left",
    left_on="CODE_CULTU",
    right_on="code"
)

print("\nEşleştirme sonrası tablo boyutu:", frh01_mapped.shape)

#Sınıf karsılıgı bulunmayan parrsellerin sayısı hesaplanır
print(
    "Eşleşmeyen satır sayısı:",
    frh01_mapped["classname"].isna().sum()
)

# Her sınıftaki parsel sayısı hesaplanır.
print("\nFRH01 sınıf dağılımı:")
print(
    frh01_mapped["classname"].value_counts()
)

eslesmeyen_urun_kodlari = frh01_mapped.loc[
    frh01_mapped["classname"].isna(),
    "CODE_CULTU"
].value_counts()

print(
    "\nEşleşmeyen farklı ürün kodu sayısı:",
    len(eslesmeyen_urun_kodlari)
)

print("\nEn sık görülen 10 eşleşmeyen ürün kodu:")
print(eslesmeyen_urun_kodlari.head(10))

# Yalnızca dokuz hedef sınıftan birine ait parseller seçilir.
frh01_temiz = frh01_mapped.dropna(
    subset=["classname"]
).copy()

print("\nAnalizde kullanılacak FRH01 boyutu:", frh01_temiz.shape)