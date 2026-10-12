# Sentinel-2 Zaman Serileriyle Tarımsal Ürün Sınıflandırması

Bu depo, **Esmanur Polattimur (241230014)** tarafından yürütülen Uygulamalı
Yapay Zekâ dönem projesinin çalıştırılabilir kodlarını içerir. Mevcut kodlar
Faz 1 veri incelemesini, TUM temel modellerini ve BreizhCrops pilot ön işleme
hattını kapsar.

## Faz 1'de kullanılan veriler

### BreizhCrops 2017 L1C

- Resmî kaynak: <https://github.com/dl4sits/BreizhCrops>
- Bölümler: FRH01, FRH02, FRH03 ve FRH04
- Girdi: değişken uzunluklu Sentinel-2 zaman serisi, 13 spektral bant
- Hedef: 9 ürün sınıfı
- Bu depodaki `frh01.csv`–`frh04.csv` dosyaları parsel indeksleridir.
- `breizhcrops_frh04_evidence/6082638.csv`, resmî FRH04 arşivinden çıkarılan
  gerçek bir arpa parselidir; 102 tarih ve 13 spektral bant içerir.

### TUM CropTypes 2016–2018

- Resmî kaynak: <https://github.com/TUM-CPN/CropTypes>
- Eğitim/geliştirme: 2.099 parsel
- Bağımsız test: 301 parsel
- Girdi: 14 zaman aralığı × 13 Sentinel-2 bandı
- Hedef: resmî eşleme sonrası 7 sınıf

İki veri setinin sınıfları ve örnekleme düzenleri aynı değildir. Bu nedenle
ham başarı skorları veri setleri arasında doğrudan karşılaştırılmamalıdır.

## Ana veri seti ve pilot çalışma

Projenin ana veri seti, `D:\Datasets\BreizhCrops\2017\L1C` altında bulunan
büyük **BreizhCrops 2017 L1C** veri setidir. TUM CropTypes yalnızca küçük bir
referans ve yöntem geliştirme veri seti olarak tutulacaktır. BreizhCrops'un
`.tar.gz` arşivlerini yeniden açmak yerine doğrudan `frh01.h5`–`frh04.h5`
dosyaları kullanılacaktır.

İlk olarak her bölge ve sınıfı temsil eden yaklaşık `%10` büyüklüğünde bir
pilot örneklem kullanılacaktır. Pilotun amacı nihai sonucu üretmek değil;
HDF5 okuma, sınıf eşleme, bulut kontrolü, zaman aralığı oluşturma, özellik
çıkarma, bellek kullanımı ve model kodunu kısa sürede doğrulamaktır. Pilot
çalışma başarılı olduğunda aynı işlem hattı örnekleme sınırı kaldırılarak tüm
uygun BreizhCrops parsellerine uygulanacaktır. Rapordaki nihai model sonuçları
pilot örneklemden değil, tam veri çalışmasından alınacaktır.

Önerilen bölge ayrımı şöyledir:

```text
FRH01 + FRH02 → eğitim
FRH03         → doğrulama ve model seçimi
FRH04         → yalnız final test
```

Pilot çalışma da bu ayrımı korur: eğitim pilotu FRH01 ve FRH02'den,
doğrulama pilotu FRH03'ten, test pilotu ise FRH04'ten alınır. Az sayıdaki
sınıfların pilot örneklemde kaybolmaması için örnekleme sınıf ve bölgeye göre
katmanlı yapılır. Büyük sınıflarda `%10` oranı kullanılır; bölge başına 20'den
az örneği bulunan sınıflarda mevcut örneklerin tümü korunur. FRH04'ün ön
işleme uyumluluğu kontrol edilir ancak pilot aşamada skoru görüntülenmez;
FRH04 üzerinde model veya ön işleme seçimi yapılmaz.

## BreizhCrops veri ön işleme kuralları

Her parsele bağımsız uygulanan sabit işlemler önce pilotta kontrol edilir,
ardından eğitim, doğrulama ve test bölümlerinin tamamına aynı şekilde
uygulanır:

1. Resmî sınıf eşlemesi uygulanır ve hedef dışındaki sınıflar ayrılır.
2. Boş zaman serileri model girdisinden çıkarılır ve sayıları raporlanır.
3. Gözlemler tarihe göre sıralanır.
4. `QA60` bilgisiyle bulutlu veya güvenilir olmayan gözlemler işaretlenir.
5. Spektral değerler uygun şekilde ölçeklenir.
6. NDVI, NDRE ve NDMI gibi spektral indeksler hesaplanır.
7. Zaman serileri ortak 15 günlük dönemlere yerleştirilir.
8. Geçerli gözlem sayısı, bulut oranı, minimum, maksimum, medyan, genlik,
   eğim, alan ve tepe zamanı gibi fenolojik özellikler çıkarılır.

Medyanla eksik değer doldurma, `StandardScaler`, özellik seçimi, PCA ve sınıf
ağırlıkları gibi veriden değer öğrenen işlemler **yalnız eğitim bölümünde**
öğrenilir. Öğrenilen değerler değiştirilmeden doğrulama ve test bölümlerine
uygulanır:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)
```

Tüm veri üzerinde `fit` işlemi yapılmaz. Aksi durumda FRH03 veya FRH04'ten
modele bilgi taşınır ve veri sızıntısı oluşur. Büyük HDF5 dosyaları da aynı
anda belleğe alınmaz; parseller parça parça okunur ve çıkarılan özellikler
önbelleğe kaydedilir.

## Klasör yapısı

```text
.
├── 241230014_Sentinel2_Faz1.ipynb # Öğrenci anlatımıyla adım adım analiz
├── 241230014_Sentinel2_Faz2.ipynb # RF ve MLP temel modelleri
├── 241230014_BreizhCrops_Pilot.ipynb # Büyük veri pilot notebooku
├── analysis/
│   ├── phase1_eda.py             # Veri denetimi, özetler ve grafikler
│   ├── build_phase1_report.py    # Analiz özetinden PDF üretimi
│   ├── phase2_models.py          # TUM için RF ve MLP temel modelleri
│   ├── phase3_breizh_preprocessing.py # Breizh ön işleme ve özellik çıkarma
│   └── phase3_breizh_rf.py       # Pilot Random Forest modeli
├── data/phase1/                  # Yerel veri dosyaları (Git'e yüklenmez)
├── breizhcrops_frh04_evidence/   # Gerçek FRH04 parsel örneği ve kanıt notu
├── output/phase1/                # Üretilen CSV, JSON, PNG ve PDF çıktıları
├── run_phase1.py                 # Tek komutlu çalıştırıcı
├── run_phase2.py                 # Faz 2 temel model çalıştırıcısı
├── run_breizh_pilot.py           # Breizh pilot çalıştırıcısı
├── run_breizh_full_rf.py         # Tam Breizh Random Forest çalıştırıcısı
└── requirements.txt              # Python bağımlılıkları
```

Analizi öğrenme ve ders sunumu amacıyla takip etmek için önce
`241230014_Sentinel2_Faz1.ipynb` açılabilir. Notebook; laboratuvar örnekleri
gibi kısa bölümlere ayrılmış, temel `pandas`, `numpy` ve `matplotlib`
komutlarıyla yazılmıştır. `analysis/phase1_eda.py` aynı yaklaşımın çıktı
dosyalarını otomatik oluşturan sürümüdür.

## Faz 2 temel modelleri

İlk Faz 2 kodu TUM CropTypes üzerinde Random Forest ve MLP modellerini
eğitir. Eğitim/geliştirme verisi %80 eğitim ve %20 doğrulama olarak ayrılır;
resmî `TestData.xlsx` dosyası yalnız bağımsız test için kullanılır.

```powershell
python run_phase2.py
```

Accuracy, Macro-F1, Balanced Accuracy, confusion matrix görselleri ve Random
Forest özellik önemleri `output/phase2/` klasörüne kaydedilir.

## BreizhCrops pilot kodunu çalıştırma

Önce her bölge ve sınıftan en fazla iki parsel alan hızlı kod testi yapılır:

```powershell
python run_breizh_pilot.py --hizli-test --uzerine-yaz
```

Bu testin başarı skorları bilimsel sonuç olarak kullanılmaz. Gerçek `%10`
pilot ön işleme ve Random Forest çalışması için:

```powershell
python run_breizh_pilot.py --oran 0.10 --uzerine-yaz
```

Yalnız özellik tablosunu oluşturmak için komuta `--sadece-on-isleme`
eklenebilir. Çıktılar `output/phase3/` altında tutulur. Pilot aşamada model
yalnız FRH03 doğrulama verisiyle değerlendirilir; FRH04 skoru gösterilmez.
Tam veri modeli ve bütün model seçimleri tamamlandıktan sonra final test
`analysis/phase3_breizh_rf.py` dosyasının `--final-testi-calistir` seçeneğiyle
yalnız bir kez hesaplanır.

## Tam BreizhCrops Random Forest

Pilot sonuçları doğrulandıktan sonra tüm eşlenmiş parselleri işlemek ve
Random Forest modelini FRH01+FRH02 üzerinde eğitip FRH03 üzerinde
değerlendirmek için:

```powershell
python run_breizh_full_rf.py
```

Tam özellik tablosu daha önce oluşturulduysa uzun ön işleme aşamasını tekrar
etmeden yalnız modeli çalıştırmak için:

```powershell
python run_breizh_full_rf.py --sadece-model
```

Bu aşamada da FRH04 skoru kapalıdır. `--final-testi-calistir` yalnız bütün
özellik ve model kararları tamamlandıktan sonra kullanılmalıdır.

## Kurulum

Windows PowerShell içinde proje klasörüne geçin ve sanal ortam oluşturun:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Çalıştırma

Önce veri dosyalarının yerinde olduğunu denetleyin:

```powershell
python run_phase1.py --check
```

Yalnız keşifsel veri analizini çalıştırmak için:

```powershell
python run_phase1.py --eda-only
```

Analiz ve PDF üretimini birlikte çalıştırmak için:

```powershell
python run_phase1.py --all
```

Parametre verilmeden `python run_phase1.py` çalıştırılması da `--all` ile aynı
işlemi yapar. Yalnız mevcut analiz özetinden PDF üretmek için
`python run_phase1.py --report-only` kullanılabilir.

## Üretilen temel çıktılar

- `output/phase1/eda_summary.json`: sayısal veri analizi özeti
- `output/phase1/breizhcrops_class_counts.csv`: BreizhCrops sınıf sayıları
- `output/phase1/tum_croptypes_class_counts.csv`: TUM sınıf sayıları
- `output/phase1/assets/`: sınıf dağılımı ve gerçek parsel grafikleri
- `output/phase1/241230014_EsmanurPolattimur_Faz1_Raporu.pdf`: Faz 1 raporu

## Bilimsel sınırlamalar

- BreizhCrops indeks dosyalarında parsel koordinatları bulunmadığından,
  geometri elde edilmeden 5/10 km mekânsal blok ayrımı yapılmış gibi
  raporlanamaz.
- TUM bağımsız test kümesinde şeker pancarı örneği yoktur; patates yalnız bir
  örnek içerir. Bu sınıflar için test F1 değeri bilimsel olarak güvenilir
  yorumlanamaz.
- Büyük veri dosyaları Git/GitHub'a yüklenmemelidir. Depoda kod, veri erişim
  bağlantıları, dosya bütünlük bilgileri ve küçük kanıt örnekleri tutulmalıdır.

## Sonraki aşama

BreizhCrops `%10` pilotu doğrulandıktan sonra aynı ön işleme kodu tam veride
çalıştırılacaktır. Ardından Random Forest, MLP, XGBoost ve 1D-CNN modelleri
aynı bölge ayrımı üzerinde eğitilecektir. Accuracy'nin yanında Macro-F1,
Balanced Accuracy, sınıf bazlı F1, NLL, Brier ve ECE ölçütleri
raporlanacaktır.
