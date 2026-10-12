from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "phase1"
ASSETS = OUT / "assets"
PDF_PATH = OUT / "241230014_EsmanurPolattimur_Faz1_Raporu.pdf"

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#2A6F77")
GOLD = colors.HexColor("#D7A84B")
LIGHT = colors.HexColor("#EAF0F4")
LIGHT_GOLD = colors.HexColor("#FBF3DF")
RED = colors.HexColor("#8B2F3C")
TEXT = colors.HexColor("#1E2A33")
MUTED = colors.HexColor("#5F6D78")
GRID = colors.HexColor("#CCD6DD")


def register_fonts() -> None:
    font_dir = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("TNR", str(font_dir / "times.ttf")))
    pdfmetrics.registerFont(TTFont("TNR-Bold", str(font_dir / "timesbd.ttf")))
    pdfmetrics.registerFont(TTFont("TNR-Italic", str(font_dir / "timesi.ttf")))
    pdfmetrics.registerFont(TTFont("TNR-BoldItalic", str(font_dir / "timesbi.ttf")))


register_fonts()

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="BodyTR", fontName="TNR", fontSize=11.2, leading=16.8,
    textColor=TEXT, alignment=TA_JUSTIFY, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="SmallTR", parent=styles["BodyTR"], fontSize=8.5, leading=11.5,
    textColor=MUTED, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="CaptionTR", parent=styles["SmallTR"], fontName="TNR-Italic",
    alignment=TA_CENTER, textColor=MUTED, spaceBefore=4, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="H1TR", fontName="TNR-Bold", fontSize=17, leading=21,
    textColor=NAVY, spaceBefore=6, spaceAfter=10, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="H2TR", fontName="TNR-Bold", fontSize=13.2, leading=17,
    textColor=TEAL, spaceBefore=8, spaceAfter=6, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="H3TR", fontName="TNR-Bold", fontSize=11.5, leading=15,
    textColor=NAVY, spaceBefore=6, spaceAfter=4, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="BulletTR", parent=styles["BodyTR"], leftIndent=14, firstLineIndent=-8,
    bulletIndent=4, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="CoverTitle", fontName="TNR-Bold", fontSize=24, leading=29,
    textColor=NAVY, alignment=TA_CENTER, spaceAfter=16,
))
styles.add(ParagraphStyle(
    name="CoverSub", fontName="TNR", fontSize=14, leading=20,
    textColor=TEAL, alignment=TA_CENTER, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="TableHead", fontName="TNR-Bold", fontSize=8.5, leading=10.5,
    textColor=colors.white, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    name="TableCell", fontName="TNR", fontSize=8.2, leading=10.2,
    textColor=TEXT, alignment=TA_LEFT,
))
styles.add(ParagraphStyle(
    name="TableCellCenter", parent=styles["TableCell"], alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    name="Callout", fontName="TNR", fontSize=10.3, leading=14.5,
    textColor=TEXT, alignment=TA_LEFT,
))
styles.add(ParagraphStyle(
    name="Ref", fontName="TNR", fontSize=8.5, leading=11.2,
    textColor=TEXT, leftIndent=15, firstLineIndent=-15, spaceAfter=4,
))


def p(text: str, style: str = "BodyTR") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(f"• {text}", styles["BulletTR"])


def h1(text: str) -> Paragraph:
    return Paragraph(text, styles["H1TR"])


def h2(text: str) -> Paragraph:
    return Paragraph(text, styles["H2TR"])


def h3(text: str) -> Paragraph:
    return Paragraph(text, styles["H3TR"])


def fmt(n: int | float) -> str:
    if isinstance(n, float) and not n.is_integer():
        return f"{n:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{int(n):,}".replace(",", ".")


def table(rows, widths, header=True, font_size=8.2, alignments=None, repeat_rows=1):
    converted = []
    for r_idx, row in enumerate(rows):
        converted_row = []
        for c_idx, cell in enumerate(row):
            if isinstance(cell, Paragraph):
                converted_row.append(cell)
            else:
                style = "TableHead" if header and r_idx == 0 else "TableCell"
                if alignments and c_idx < len(alignments) and alignments[c_idx] == "C":
                    style = "TableHead" if header and r_idx == 0 else "TableCellCenter"
                converted_row.append(Paragraph(str(cell), styles[style]))
        converted.append(converted_row)
    t = Table(converted, colWidths=widths, repeatRows=repeat_rows if header else 0, hAlign="LEFT")
    ts = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.35, GRID),
        ("FONTNAME", (0, 1 if header else 0), (-1, -1), "TNR"),
        ("FONTSIZE", (0, 1 if header else 0), (-1, -1), font_size),
    ]
    if header:
        ts += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
        ]
    for idx in range(1 if header else 0, len(rows)):
        if idx % 2 == 0:
            ts.append(("BACKGROUND", (0, idx), (-1, idx), LIGHT))
    t.setStyle(TableStyle(ts))
    return t


def callout(title: str, text: str, warning: bool = False):
    bg = colors.HexColor("#FBEAEC") if warning else LIGHT_GOLD
    bar = RED if warning else GOLD
    content = Paragraph(f"<b>{title}</b><br/>{text}", styles["Callout"])
    t = Table([[content]], colWidths=[17.1 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.6, bar),
        ("LINEBEFORE", (0, 0), (0, -1), 5, bar),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def image_with_caption(path: Path, caption: str, width=16.8 * cm):
    img = Image(str(path))
    img._restrictSize(width, 8.3 * cm)
    return KeepTogether([img, p(caption, "CaptionTR")])


class ReportDoc(BaseDocTemplate):
    pass


def page_decor(canvas, doc):
    canvas.saveState()
    w, h = A4
    if doc.page > 1:
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(2.1 * cm, h - 1.35 * cm, w - 2.1 * cm, h - 1.35 * cm)
        canvas.setFont("TNR", 8.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(2.1 * cm, h - 1.05 * cm, "Uygulamalı Yapay Zeka — Proje Faz 1")
        canvas.drawRightString(w - 2.1 * cm, h - 1.05 * cm, "241230014 — Esmanur Polattimur")
    canvas.setFont("TNR", 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(w / 2, 1.0 * cm, f"Sayfa {doc.page}")
    canvas.restoreState()


def build() -> None:
    data = json.loads((OUT / "eda_summary.json").read_text(encoding="utf-8"))
    b = data["breizhcrops"]
    t = data["tum_croptypes"]

    doc = ReportDoc(
        str(PDF_PATH), pagesize=A4,
        leftMargin=2.1 * cm, rightMargin=2.1 * cm,
        topMargin=1.8 * cm, bottomMargin=1.6 * cm,
        title="Uygulamalı Yapay Zeka Proje Faz 1 Raporu",
        author="Esmanur Polattimur",
        subject="Sentinel-2 uydu zaman serileri ile ürün sınıflandırma",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=page_decor)])

    story = []

    # Cover
    story += [Spacer(1, 2.0 * cm)]
    story += [p("YAPAY ZEKA VE MAKİNE ÖĞRENMESİ BÖLÜMÜ", "CoverSub")]
    story += [p("UYGULAMALI YAPAY ZEKA DERSİ", "CoverSub")]
    story += [Spacer(1, 1.2 * cm)]
    story += [p("PROJE FAZ 1 İLERLEME RAPORU", "CoverTitle")]
    story += [p(
        "Coğrafi Dağılım Değişiminde Sentinel‑2 Zaman Serileri ile Ürün Sınıflandırma:<br/>"
        "Performans, Güven Kalibrasyonu ve Yüksek Güvenli Hatalar",
        "CoverSub",
    )]
    story += [Spacer(1, 1.7 * cm)]
    cover_rows = [
        ["Öğrenci", "Esmanur Polattimur"],
        ["Öğrenci numarası", "241230014"],
        ["Proje türü", "Bireysel"],
        ["Rapor tarihi", "3 Ekim 2026"],
        ["İncelenen veri setleri", "BreizhCrops 2017 L1C ve TUM CropTypes 2016–2018"],
    ]
    ct = table(cover_rows, [5.0 * cm, 9.0 * cm], header=False, alignments=["C", "C"], repeat_rows=0)
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), NAVY),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("FONTNAME", (0, 0), (0, -1), "TNR-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.8, NAVY),
        ("LINEBELOW", (0, 0), (-1, -2), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [ct, Spacer(1, 1.7 * cm)]
    story += [callout(
        "Raporun konumu",
        "Bu belge model sonuç raporu değildir. Faz 1 kapsamında veri kaynaklarının doğrulanması, gerçek dosyalardan keşifsel veri analizi, ön işleme kararları, literatür sentezi ve kullanılacak dört modelin belirlenmesi raporlanmıştır.",
    )]
    story += [PageBreak()]

    # Executive summary and compliance
    story += [h1("1. Faz 1 kapsamı ve ulaşılan sonuç")]
    story += [p(
        "Ders tebliğatı Faz 1 için veri setlerinin bulunmasını ve görselleştirilerek analiz edilmesini; veri sayısı, sınıf dağılımı, eksik değer ve etiket durumunun gösterilmesini; gerekli ön işlemlerin tanımlanmasını; literatür sonuçlarının özetlenmesini ve kullanılacak mimarilerin seçilmesini istemektedir. Bu rapor, söz konusu dört başlığı gerçek veri dosyaları üzerinden karşılar."
    )]
    story += [p(
        "Ana veri seti <b>BreizhCrops 2017 L1C</b> olarak korunmuştur. İkinci veri seti, onay öncesi kapsam düzeltmesiyle <b>TUM CropTypes 2016–2018</b> olarak belirlenmiştir. BavarianCrops arşivi 322.932.645 bayt olduğundan Faz 1'de yerel dosya denetimi tamamlanamamış; yalnızca kaynağa atıf yapıp veriyi incelenmiş gibi göstermek yerine, tüm dosyaları indirilebilen ve denetlenebilen TUM CropTypes seçilmiştir. Bu değişiklik öğretim elemanı onayından önce bildirilmelidir."
    )]
    story += [callout(
        "Kısa hüküm",
        "Proje yapılabilir; fakat iki veri setinin sınıf uzayı ve zaman ekseni aynı değildir. Bu nedenle veri setleri arasında ham başarı puanı yarıştırılmayacak, her veri setinde aynı model aileleri ayrı ayrı değerlendirilecektir. TUM bağımsız testinin sınıf desteği zayıf olduğundan ana bilimsel sonuç BreizhCrops üzerinden kurulacaktır.",
        warning=True,
    ), Spacer(1, 8)]
    compliance_rows = [
        ["Tebliğat maddesi", "Bu rapordaki somut karşılık", "Durum"],
        ["En az 2 gerçek veri seti ve bağlantıları", "BreizhCrops + TUM CropTypes; resmî depo ve dosya bağlantıları", "Tamamlandı"],
        ["Veri sayısı, sınıf sayıları, eksik değer, etiket", "767.504/608.280 Breizh kaydı; 2.099+301 TUM kaydı; sınıf ve eksiklik tabloları", "Tamamlandı"],
        ["Görselleştirme", "Sınıf dağılımları ve iki gerçek parsel zaman serisi", "Tamamlandı"],
        ["Ön işleme", "Etiket eşleme, boş seri eleme, zaman hizalama, ölçekleme ve maske protokolü", "Protokol belirlendi"],
        ["Literatür analizi", "7 çalışma: veri, yöntem, sonuç ve sınırlılık", "Tamamlandı"],
        ["3–5 model/mimari", "RF, XGBoost, MLP (negatif kontrol), 1D‑CNN", "Belirlendi"],
        ["Git ile sürüm kontrolü", "Yerel çalışma dosyaları var; öğrenciye ait GitHub deposu henüz yok", "Eksik"],
    ]
    story += [table(compliance_rows, [4.0 * cm, 10.0 * cm, 2.8 * cm], alignments=["L", "L", "C"])]
    story += [PageBreak()]

    # Problem and sources
    story += [h1("2. Problem tanımı, araştırma sorusu ve veri kaynakları")]
    story += [h2("2.1 Problem tanımı")]
    story += [p(
        "Sentinel‑2 görüntülerinden bir tarım parseli için yıl boyunca elde edilen çok bantlı gözlemler, ürünün fenolojik gelişimini temsil eden çok değişkenli bir zaman serisi oluşturur. Amaç, bu zaman serisini kullanarak her parselin ürün sınıfını tahmin etmektir. Eğitim ve test bölgeleri farklı olduğunda iklim, toprak, ekim takvimi, bulutluluk ve sınıf oranları değişebilir. Bu coğrafi dağılım değişimi yalnız doğruluğu değil, model olasılıklarının güvenilirliğini de düşürebilir."
    )]
    story += [p(
        "<b>Araştırma sorusu:</b> Random Forest, XGBoost, zaman sırasını doğrudan kullanmayan MLP ve 1D‑CNN; kaynak bölge içi tutulan örneklerden farklı coğrafi bölgelere geçildiğinde Macro‑F1, kalibrasyon ve yüksek güvenli hata bakımından nasıl değişmektedir?"
    )]
    story += [callout(
        "Uydu verisi neden fotoğraf gibi görünmüyor?",
        "Kaynak Sentinel‑2 görüntüleridir; ancak bu veri setlerinde her parselin pikselleri tarih ve bant bazında ortalanarak sayısal zaman serisine dönüştürülmüştür. Dolayısıyla model girdisi RGB fotoğraf değil, her tarih için 13 spektral bant değerinden oluşan bir matristir. Görsel üretmek mümkündür; fakat veri dosyasının asıl biçimi tablo/zaman serisidir.",
    )]
    story += [h2("2.2 Veri setleri ve resmî erişim")]
    source_rows = [
        ["Veri seti", "Sürüm ve kapsam", "Yerel olarak doğrulanan dosyalar", "Resmî erişim"],
        [
            "BreizhCrops",
            "2017, Sentinel‑2 L1C, Brittany/Fransa, FRH01–FRH04, parsel zaman serisi",
            "classmapping.csv ve dört bölge indeks dosyası; ayrıca gerçek FRH04 parseli 6082638",
            p('<link href="https://github.com/dl4sits/BreizhCrops">github.com/dl4sits/BreizhCrops</link>', "TableCell"),
        ],
        [
            "TUM CropTypes",
            "2016–2018, Sentinel‑2, Yukarı Bavyera/Almanya, iki haftalık parsel özetleri",
            "data2016-2018.xlsx (2.099 satır) ve TestData.xlsx (301 satır)",
            p('<link href="https://github.com/TUM-CPN/CropTypes">github.com/TUM-CPN/CropTypes</link>', "TableCell"),
        ],
    ]
    story += [table(source_rows, [2.7 * cm, 4.6 * cm, 5.8 * cm, 3.8 * cm])]
    story += [p(
        "Dosyalar 3 Ekim 2026 tarihinde resmî kaynaklardan indirilmiştir. Tekrarlanabilirlik için dosya boyutları ve SHA‑256 özetleri Ek A'da verilmiştir."
    )]
    story += [h2("2.3 Kapsam kararı")]
    story += [bullet("BreizhCrops ana deneydir: veri hacmi, dört coğrafi bölge ve aşırı sınıf dengesizliği araştırma sorusuna uygundur.")]
    story += [bullet("TUM CropTypes ikincil deneydir: farklı yıl ve coğrafyada aynı Sentinel‑2/parsel sınıflandırma problemine küçük ölçekli bir stres testi sağlar.")]
    story += [bullet("İki veri setinin sınıf adları bire bir eşit değildir. Ortak birleştirilmiş etiket uzayı oluşturulmayacak; modeller her veri setinde ayrı eğitilecektir.")]
    story += [bullet("BavarianCrops Faz 1 kapsamından çıkarılmıştır. Büyük arşiv ve tamamlanmamış yerel denetim nedeniyle raporda analiz edilmiş gibi sunulması bilimsel olarak savunulamaz.")]
    story += [PageBreak()]

    # Breizh EDA
    story += [h1("3. Veri seti 1 — BreizhCrops keşifsel analizi")]
    story += [h2("3.1 Dosya ve kolon yapısı")]
    story += [p(
        "Bölge indeks dosyalarının kolonları <b>id, CODE_CULTU, path, idx, meanQA60, sequencelength</b> biçimindedir. <i>classmapping.csv</i>, 23 ham ürün kodunu 0–8 arasındaki dokuz hedef sınıfa eşler. İndeks satırı bir parseli, <i>path</i> alanı ise o parselin tarih-bant gözlemlerini içeren ayrı CSV dosyasını gösterir. Gerçek parsel dosyasında 13 bant (B1–B12 ve B8A), QA10/QA20/QA60, tarih, ham etiket ve parsel kimliği bulunur."
    )]
    region_rows = [["Bölge", "Ham parsel", "9 sınıfa eşlenen", "Eşleme dışı", "Medyan seri", "Sıfır uzunluk (eşlenen)"]]
    zeros = {"FRH01": 4, "FRH02": 4, "FRH03": 6, "FRH04": 3}
    for region, info in b["regions"].items():
        region_rows.append([
            region, fmt(info["raw_rows"]), fmt(info["mapped_rows"]), fmt(info["unmapped_rows"]),
            fmt(info["sequence_length"]["median"]), fmt(zeros[region]),
        ])
    region_rows.append(["Toplam", fmt(b["raw_total_rows"]), fmt(b["mapped_total_rows"]), fmt(b["unmapped_total_rows"]), "—", "17"])
    story += [table(region_rows, [2.2 * cm, 2.8 * cm, 3.0 * cm, 2.8 * cm, 2.7 * cm, 3.2 * cm], alignments=["C"] * 6)]
    story += [p(
        "Ham indekslerde 63 <i>meanQA60</i> eksikliği vardır; dokuz sınıfa eşlenen alt kümede bu sayı 17'dir ve bunların tamamı sıfır uzunluklu serilerle çakışmaktadır. Kimliklerde yinelenen kayıt bulunmamıştır. Eşleme dışındaki 159.224 kayıt hedef problemden çıkarılacaktır; bu satırlar 'eksik etiket' değil, hedef sınıf sözlüğü dışında kalan ürün kodlarıdır."
    )]
    story += [image_with_caption(
        ASSETS / "breizhcrops_class_distribution.png",
        "Şekil 1. Resmî indeks ve sınıf eşleme dosyalarından hesaplanan gerçek sınıf dağılımı. Sınıf dengesizliği belirgindir; ayçiçeği 19, kabuklu yemiş 49 parsel ile aşırı az temsil edilmektedir.",
    )]
    story += [PageBreak()]

    story += [h2("3.2 Sınıf dağılımı")]
    bc = b["class_counts"]
    class_rows = [
        ["Sınıf", "FRH01", "FRH02", "FRH03", "FRH04", "Toplam"],
        ["Arpa", "13.046", "10.733", "7.148", "5.978", fmt(bc["barley"])],
        ["Buğday", "30.368", "15.005", "27.190", "16.993", fmt(bc["wheat"])],
        ["Kolza", "5.593", "2.347", "3.557", "3.236", fmt(bc["rapeseed"])],
        ["Mısır", "43.990", "36.593", "41.993", "31.333", fmt(bc["corn"])],
        ["Ayçiçeği", "1", "6", "10", "2", fmt(bc["sunflower"])],
        ["Meyve bahçeleri", "944", "350", "1.224", "553", fmt(bc["orchards"])],
        ["Kabuklu yemiş", "10", "18", "10", "11", fmt(bc["nuts"])],
        ["Kalıcı çayır", "32.651", "36.513", "32.534", "26.118", fmt(bc["permanent meadows"])],
        ["Geçici çayır", "52.014", "39.084", "52.731", "38.393", fmt(bc["temporary meadows"])],
    ]
    story += [table(class_rows, [4.0 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.6 * cm], alignments=["L", "C", "C", "C", "C", "C"])]
    story += [callout(
        "Kritik veri sorunu",
        "Ayçiçeği ve kabuklu yemiş sınıfları için standart bir sınıflandırma skoru istatistiksel olarak güvenilir değildir. Sınıf ağırlığı kullanmak bu veri yokluğunu gidermez. Ana sonuçlar dokuz sınıfla raporlanacak; ancak bu iki sınıf için örnek sayısı ve geniş belirsizlik ayrıca gösterilecek, gerekirse önceden tanımlı duyarlılık analizinde çıkarılacaktır.",
        warning=True,
    )]
    story += [h2("3.3 Gerçek parsel örneği ve X/y biçimi")]
    story += [p(
        "FRH04 bölgesinden <b>6082638</b> numaralı gerçek parsel 102 tarih gözlemi içerir. Ham etiketi <b>ORH</b>, resmî eşlemedeki sınıfı <b>arpa</b> ve sınıf kimliği <b>0</b>'dır. Tarih aralığı 02.01.2017–28.12.2017'dir; incelenen dosyada eksik hücre yoktur. Ham girdi <b>X ∈ R<super>102×13</super></b>, hedef <b>y = 0</b> biçimindedir."
    )]
    sample = b["sample"]["first_three_records"]
    sample_rows = [["Tarih", "B2", "B3", "B4", "B8", "QA60", "Etiket"]]
    for r in sample:
        sample_rows.append([
            r["doa"], f'{r["B2"]:.2f}', f'{r["B3"]:.2f}', f'{r["B4"]:.2f}',
            f'{r["B8"]:.2f}', f'{r["QA60"]:.0f}', r["label"],
        ])
    story += [table(sample_rows, [2.8 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.0 * cm], alignments=["C"] * 7)]
    story += [image_with_caption(
        ASSETS / "breizhcrops_real_parcel_6082638.png",
        "Şekil 2. FRH04/6082638 parselinin gerçek B2, B3, B4 ve B8 zaman serisi. Değerler görselleştirmede 10⁻⁴ ile ölçeklenmiştir; dosyadaki sayılar değiştirilmemiştir.",
    )]
    story += [PageBreak()]

    # TUM EDA
    story += [h1("4. Veri seti 2 — TUM CropTypes keşifsel analizi")]
    story += [h2("4.1 Dosya ve özellik yapısı")]
    story += [p(
        "Resmî depodaki <i>data2016-2018.xlsx</i> dosyası 2.099 parsel ve 229 kaynak kolonu; <i>TestData.xlsx</i> dosyası 301 parsel ve 230 kaynak kolonu içerir. Eğitim/geliştirme kayıtları 2016: 699, 2017: 700 ve 2018: 700 biçimindedir; bağımsız test yalnız 2018 yılına aittir. Her parsel Şubat–Ağustos arasındaki 14 yarım-ay aralığında 13 Sentinel‑2 bandı ve NDVI, NDRE, REIP olmak üzere 16 değişkenle temsil edilir. Böylece 224 sayısal zaman özelliği vardır."
    )]
    story += [p(
        "Her iki dosyada da eksik hücre, sonsuz sayı veya tamamen yinelenen satır bulunmamıştır. Yalnız spektral bantlar kullanılırsa 1D‑CNN girdileri eğitim için <b>(2.099, 14, 13)</b>, test için <b>(301, 14, 13)</b>; RF/XGBoost/MLP girdileri sırasıyla <b>(2.099, 182)</b> ve <b>(301, 182)</b> olur. Hedef vektörleri <b>y ∈ R<super>2099</super></b> ve <b>y ∈ R<super>301</super></b> biçimindedir."
    )]
    story += [h2("4.2 Resmî sınıf eşlemesi ve dağılım")]
    tc = t["split_class_counts"]
    tum_rows = [["Sınıf", "Eğitim/geliştirme", "Bağımsız test", "Toplam"]]
    for name in ["patates", "kışlık arpa", "mısır", "kışlık kolza", "kışlık buğday", "şeker pancarı", "diğer"]:
        tum_rows.append([name.title(), fmt(tc["Eğitim/geliştirme"][name]), fmt(tc["Bağımsız test"][name]), fmt(t["class_counts"][name])])
    story += [table(tum_rows, [5.0 * cm, 4.2 * cm, 4.0 * cm, 3.5 * cm], alignments=["L", "C", "C", "C"])]
    story += [image_with_caption(
        ASSETS / "tum_croptypes_class_distribution.png",
        "Şekil 3. Resmî çalışma defterindeki eşleme kuralı uygulandıktan sonraki gerçek sınıf dağılımı. Eğitim kümesi dengeli; bağımsız test kümesi ciddi biçimde dengesizdir.",
    )]
    story += [callout(
        "Test kümesi sorunu",
        "Bağımsız testte şeker pancarı örneği yoktur ve patates yalnız bir örnektir. Yedi sınıflı Macro‑F1 bu koşulda yanıltıcıdır. TUM sonuçlarında sınıf desteği, confusion matrix ve desteklenen sınıflar üzerindeki duyarlılık analizi zorunlu tutulacaktır; bu veri seti ana kıyas değil ikincil stres testi olacaktır.",
        warning=True,
    )]
    story += [h2("4.3 Gerçek parsel örneği")]
    ts = t["sample"]
    story += [p(
        f"Eğitim dosyasındaki {ts['row_index']} numaralı gerçek satır 2017 yılına, ham kod <b>{ts['raw_code']}</b> ve "
        f"<b>{ts['raw_label']}</b> etiketine aittir; eşlenen sınıf <b>{ts['mapped_class']}</b>'dır. "
        "Aşağıdaki değerler yarım-ay aralıklarında parsel ortalaması olarak saklanan gerçek bant değerleridir."
    )]
    selected = ts["selected_records"]
    keep_bins = {"feb1", "march1", "april1", "may1", "june1", "july1", "august2"}
    tum_sample_rows = [["Zaman aralığı", "B2", "B3", "B4", "B8"]]
    for r in selected:
        if r["time_bin"] in keep_bins:
            tum_sample_rows.append([r["time_bin"], f'{r["B2"]:.2f}', f'{r["B3"]:.2f}', f'{r["B4"]:.2f}', f'{r["B8"]:.2f}'])
    story += [table(tum_sample_rows, [4.4 * cm, 3.1 * cm, 3.1 * cm, 3.1 * cm, 3.1 * cm], alignments=["C"] * 5)]
    story += [image_with_caption(
        ASSETS / "tum_croptypes_real_parcel.png",
        "Şekil 4. TUM CropTypes eğitim dosyasından gerçek bir kışlık buğday parselinin B2, B3, B4 ve B8 zaman örüntüsü. Grafik ekseni yarım-ay aralıklarını yaklaşık tarihle gösterir; dosyada kesin çekim tarihi değil dönem etiketi vardır.",
    )]
    story += [h2("4.4 İki veri setinin karşılaştırılması")]
    compare_rows = [
        ["Özellik", "BreizhCrops", "TUM CropTypes"],
        ["Coğrafya / yıl", "Brittany, Fransa / 2017", "Yukarı Bavyera, Almanya / 2016–2018"],
        ["Birim", "Parsel zaman serisi", "Parsel zaman serisi"],
        ["Zaman uzunluğu", "Değişken, 0–104; medyan bölgeye göre 50–102", "Sabit 14 yarım-ay aralığı"],
        ["Spektral kanal", "13 Sentinel‑2 L1C bandı", "13 bant + NDVI/NDRE/REIP"],
        ["Hedef sınıf", "9 sınıf; 608.280 eşlenen parsel", "7 sınıf; 2.099 eğitim + 301 test"],
        ["Ana risk", "Aşırı nadir sınıflar ve coğrafi dağılım farkı", "Testte eksik/tek örnekli sınıflar"],
    ]
    story += [table(compare_rows, [3.7 * cm, 6.6 * cm, 6.6 * cm])]
    story += [PageBreak()]

    # Preprocessing
    story += [h1("5. Ön işleme ve kullanılabilir veri hattı")]
    story += [h2("5.1 Faz 1'de tamamlanan işlemler")]
    for text in [
        "Resmî dosyalar indirildi; boyut ve SHA‑256 değerleri kaydedildi.",
        "BreizhCrops ham kodları resmî sınıf sözlüğüyle eşlendi; eşleme dışındaki kayıtlar ayrı sayıldı.",
        "TUM ham NC kodları resmî çalışma defterindeki kurala göre altı ürün sınıfı ve 'diğer' sınıfına eşlendi.",
        "Eksik hücre, yinelenen kimlik/satır, sonlu olmayan değer ve sıfır uzunluk denetimleri yapıldı.",
        "Her veri setinden gerçek parsel örneği okunarak zaman×bant matris biçimi doğrulandı.",
        "Sınıf dağılımı ve gerçek bant zaman serisi görselleri üretildi.",
    ]:
        story += [bullet(text)]
    story += [h2("5.2 Faz 2 öncesi uygulanacak kesin protokol")]
    prep_rows = [
        ["Adım", "BreizhCrops", "TUM CropTypes", "Sızıntı önlemi"],
        ["Etiket", "Yalnız resmî 9 sınıf; eşleme dışı kayıtlar çıkarılır", "Resmî 7 sınıf eşlemesi; ham kod korunur", "Eşleme tüm bölünmelerden bağımsız ve sabittir"],
        ["Geçersiz seri", "17 sıfır uzunluklu eşlenmiş parsel çıkarılır", "Eksik/sonlu olmayan değer yok", "Kural model seçmeden önce dondurulur"],
        ["Zaman ekseni", "10 günlük 36 sabit aralık; aralık medyanı", "Mevcut 14 yarım-ay aralığı", "Aynı veri setindeki tüm modeller aynı ekseni kullanır"],
        ["Boş aralık", "≤30 gün iki komşu arasında doğrusal enterpolasyon; aksi halde eğitim medyanı + maske", "Dosya önceden enterpole; sabit değer dizileri ayrıca işaretlenir", "Medyan yalnız eğitimden hesaplanır"],
        ["Ölçek", "Bant bazında eğitim medyanı/IQR veya standartlaştırma", "Aynı", "Test istatistiği kullanılmaz"],
        ["Girdi", "36×13 (+36 eksiklik maskesi)", "14×13; indeksler ayrı ablasyon", "Model ailesi içinde aynı özellikler"],
    ]
    story += [table(prep_rows, [2.3 * cm, 5.1 * cm, 4.6 * cm, 4.8 * cm], font_size=7.6)]
    story += [callout(
        "Düzeltilen metodoloji hatası",
        "Önceki öneride FRH01–FRH02 içinde 5/10 km mekânsal bloklar vaat edilmişti; indirilen indeks dosyalarında koordinat yoktur. Geometri alınmadan bu vaat uygulanamaz. Faz 2'de iki seçenekten biri yazılı olarak dondurulacaktır: (1) resmî parsel geometrisini edinip blok bölme, veya (2) mekânsal blok iddiasından vazgeçip 'kaynak-bölge içi tutulan örnek' ifadesini kullanma. Koordinat yokken blok üretilmiş gibi rapor yazılmayacaktır.",
        warning=True,
    )]
    story += [h2("5.3 Önerilen veri bölünmesi")]
    split_rows = [
        ["Küme", "BreizhCrops", "Amaç"],
        ["Eğitim", "FRH01 + FRH02'nin yalnız eğitim bölümü", "Model parametreleri ve ölçek istatistikleri"],
        ["Doğrulama", "Kaynak bölgelerden ayrı tutulan parseller", "Hiperparametre / erken durdurma"],
        ["Kalibrasyon", "Doğrulamadan ayrı kaynak parseller", "Sıcaklık T ve güven eşiği"],
        ["OOD geliştirme", "FRH03", "Protokol denetimi; ardından kararlar dondurulur"],
        ["OOD nihai test", "FRH04", "Tek seferlik nihai coğrafi test"],
        ["İkincil test", "TUM bağımsız 301 parsel", "Genelleme stres testi; ham skorlar Breizh ile doğrudan kıyaslanmaz"],
    ]
    story += [table(split_rows, [3.2 * cm, 6.3 * cm, 7.4 * cm])]
    story += [PageBreak()]

    # Literature
    story += [h1("6. Literatür taraması ve yöntem sentezi")]
    story += [p(
        "Literatürün ortak sonucu, zaman serisi modelinden önce veri kalitesi ve bölünme protokolünün belirleyici olduğudur. Aşağıdaki özet yalnız yöntem adlarını sıralamaz; çalışma verisi, ana bulgu ve bu proje açısından sınırlılığı birlikte gösterir."
    )]
    lit_rows = [
        ["Kaynak", "Veri / yöntem", "Ana bulgu", "Bu proje için çıkarım veya sınırlılık"],
        ["Rußwurm vd. (2020) [1]", "BreizhCrops; RF ve yedi derin zaman serisi modeli", "Geniş ölçekli, bölgesel benchmark sağlandı.", "FRH03 üzerinde hiperparametre seçiminin L1C lehine yanlılık oluşturabileceği yazarlarca belirtilir."],
        ["Rußwurm & Körner (2020) [2]", "Ham optik zaman serileri; self-attention, RNN, CNN", "Ön işleme tüm modelleri iyileştirdi; ham seride attention/RNN, CNN'den güçlüydü.", "Model mimarisinden önce veri hattı ve eksik gözlem işlemi dondurulmalıdır."],
        ["Martini vd. (2021) [3]", "Çok bölgeli Sentinel‑2; self-attention + domain-adversarial eğitim", "Coğrafi spektral kayma için alan-uyarlama yaklaşımı incelendi.", "Bu proje önce uyarlama yapmadan kaybı ve kalibrasyon bozulmasını ölçecek."],
        ["Rußwurm vd. (2023) [4]", "BreizhCrops, BavarianCrops ve iki Afrika veri seti; erken sınıflandırma", "ELECTS doğruluk‑erkenlik ödünleşimini uçtan uca öğrenir.", "Veri setleri arası sınıf/destek farkı ham skor kıyasını sınırlar."],
        ["Yuan vd. (2023) [5]", "BreizhCrops/BavarianCrops; RF tabanlı erken sınıflandırma ve özellik önemi", "Klasik modelin açıklanabilir güçlü bir karşılaştırma noktası olabileceğini gösterir.", "Alt örnekleme ve seçilen sınıf desteği literatür skoruyla bire bir eşleşmeyi engeller."],
        ["Antonijević vd. (2023) [6]", "BreizhCrops → Vojvodina; Transformer tabanlı transfer", "Kısıtlı hedef etiketiyle kısmi ince ayar yararlı bulundu.", "Bu proje transfer öğrenmeden önce güven bozulmasını temel çizgi olarak ölçeyecek."],
        ["Marszalek vd. (2022) [7]", "TUM CropTypes; RF ve SVM, ham bant/NDVI, iklim anomalisi", "Ham bantlarla RF en iyi; OA %92'ye, bilinmeyen yıl/testte %86'ya kadar bildirildi.", "Ön baskıdır; bağımsız testte sınıf desteği bizim dosya denetimimizde ciddi dengesizlik gösterir."],
    ]
    story += [table(lit_rows, [2.8 * cm, 4.3 * cm, 4.7 * cm, 5.1 * cm], font_size=7.2)]
    story += [h2("6.1 Literatürden çıkan deney ilkeleri")]
    for text in [
        "Model aileleri kadar ön işleme önemlidir; tüm modeller aynı parsel, bant ve zaman eksenini kullanmalıdır.",
        "Rastgele parsel bölmesi coğrafi genellemeyi olduğundan iyi gösterebilir; bölgesel test ayrı tutulmalıdır.",
        "Sadece accuracy sınıf dengesizliğini gizler; Macro‑F1 ve sınıf bazlı destek zorunludur.",
        "Olasılık güveni yüksek accuracy ile aynı kavram değildir; ECE, Brier ve NLL ayrıca ölçülmelidir.",
        "TUM için literatürde bildirilen OA, bu çalışmanın farklı sınıf/eşleme protokolüne doğrudan hedef değer olarak alınmamalıdır.",
    ]:
        story += [bullet(text)]
    story += [PageBreak()]

    # Models and evaluation
    story += [h1("7. Seçilen model ve değerlendirme planı")]
    model_rows = [
        ["Model", "Girdi", "Neden seçildi?", "Uygunluk"],
        ["Random Forest", "Düzleştirilmiş zaman×bant", "Küçük/orta veride güçlü, ölçeklemeye az duyarlı, literatür karşılaştırması var", "Uygun klasik temel model"],
        ["XGBoost", "Düzleştirilmiş zaman×bant", "Doğrusal olmayan etkileşimleri artırmalı ağaçlarla yakalar", "Uygun klasik rakip"],
        ["MLP", "Düzleştirilmiş zaman×bant", "Aynı sayısal girdiyi kullanır fakat yerel zaman komşuluğunu açıkça modellemez", "Bilinçli olarak zayıf/uygunsuz negatif kontrol"],
        ["1D‑CNN", "Zaman×kanal tensörü", "Fenolojik yerel örüntüleri ortak çekirdeklerle öğrenir", "Uygun zamansal model"],
    ]
    story += [table(model_rows, [2.8 * cm, 3.5 * cm, 7.0 * cm, 3.6 * cm])]
    story += [p(
        "Bu dört model, tebliğattaki en az üç–en fazla beş mimari koşulunu karşılar ve 'uygun olmayan' yaklaşımı belirsiz bırakmaz. MLP, zaman sırasını mimari olarak kullanmayan negatif kontroldür. Transformer eklenmemiştir; veri hacmi ve hesap maliyeti göz önüne alındığında dört model mevcut araştırma sorusu için yeterlidir."
    )]
    story += [h2("7.1 Adil karşılaştırma")]
    for text in [
        "Her model aynı eğitim/doğrulama/kalibrasyon/test parsellerini kullanacaktır.",
        "Hiperparametre araması model başına önceden tanımlı en fazla 20 deneme ile sınırlandırılacaktır.",
        "Nihai yapılandırmalar beş rastgele başlangıçla çalıştırılacak; ortalama ve standart sapma verilecektir.",
        "Sınıf ağırlıkları ve ölçek parametreleri yalnız eğitim bölümünden hesaplanacaktır.",
        "FRH03 incelemesinden sonra protokol dondurulacak; FRH04 seçim veya eşik ayarı için kullanılmayacaktır.",
    ]:
        story += [bullet(text)]
    story += [h2("7.2 Ölçütler ve görseller")]
    metric_rows = [
        ["Boyut", "Ölçüt / çıktı", "Gerekçe"],
        ["Sınıflandırma", "Accuracy, Macro‑F1, Balanced Accuracy, sınıf bazlı precision/recall/F1", "Dengesiz sınıflarda tek accuracy yeterli değildir"],
        ["Hata yapısı", "Ham ve normalize confusion matrix", "Hangi sınıfların karıştığını gösterir"],
        ["Kalibrasyon", "NLL, çok sınıflı Brier, 15 eşit frekanslı top-label ECE", "Olasılıkların güvenilirliğini ölçer"],
        ["Yüksek güvenli hata", "τ=0,80/0,90/0,95 için HCER, hata sayısı ve kapsama", "Yanlış ama emin tahminleri ayırır"],
        ["Dağılım farkı", "ΔMacro‑F1, ΔECE, ΔHCER; sınıf oranı JS uzaklığı", "Kaynak‑hedef bozulmasını nicelleştirir"],
        ["Belirsizlik", "5 başlangıç + 1.000 sınıf-katmanlı bootstrap %95 GA", "Tek koşu sonucuna aşırı yorum yapılmasını önler"],
    ]
    story += [table(metric_rows, [3.3 * cm, 7.6 * cm, 6.0 * cm])]
    story += [PageBreak()]

    # Status and risks
    story += [h1("8. Faz 1 durumu, açık riskler ve sonraki adımlar")]
    status_rows = [
        ["İş kalemi", "Somut çıktı", "Durum"],
        ["Veri erişimi", "Breizh indeksleri, gerçek FRH04 parseli, iki TUM Excel dosyası", "Tamam"],
        ["Veri analizi", "Satır/sınıf/eksik değer/tekrar/zaman-boyut denetimi", "Tamam"],
        ["Görselleştirme", "4 şekil", "Tamam"],
        ["Ön işleme tasarımı", "Etiket, zaman, boşluk, ölçek ve maske kuralları", "Tanımlı"],
        ["Literatür", "7 kaynaklık yöntem-sonuç-sınırlılık sentezi", "Tamam"],
        ["Mimari seçimi", "RF, XGBoost, MLP, 1D‑CNN", "Tamam"],
        ["Mekânsal blok geometrisi", "Breizh indeksinde koordinat yok", "Çözülmedi"],
        ["GitHub deposu", "Öğrenci hesabında proje deposu", "Oluşturulmadı"],
        ["Model eğitimi", "Faz 2 işi", "Başlamadı"],
    ]
    story += [table(status_rows, [4.0 * cm, 8.8 * cm, 4.1 * cm], alignments=["L", "L", "C"])]
    story += [h2("8.1 Öncelik sırasına göre yapılacaklar")]
    next_steps = [
        "Öğretim elemanına ikinci veri setinin BavarianCrops yerine TUM CropTypes olarak revize edildiği ve gerekçesi açıkça sunulacak.",
        "Öğrenciye ait GitHub deposu açılacak; veri dosyaları değil, indirme/analiz betikleri, ortam dosyası ve README sürümlenecek.",
        "Breizh parsel geometrisi bulunabiliyorsa 10 km blok protokolü uygulanacak; bulunamıyorsa rapordaki 'mekânsal blok' iddiası tamamen kaldırılacak.",
        "Ön işleme veri sızıntısı testleriyle kodlanacak ve küçük alt kümede bellek/süre ölçülecek.",
        "Önce RF ve MLP temel modelleri, sonra XGBoost ve 1D‑CNN eğitilecek.",
        "Kalibrasyon bölümü ayrılacak; FRH04'e dokunmadan FRH03 sonrası protokol dondurulacak.",
    ]
    for i, text in enumerate(next_steps, 1):
        story += [p(f"<b>{i}.</b> {text}", "BulletTR")]
    story += [h2("8.2 Gerçekçi risk değerlendirmesi")]
    risk_rows = [
        ["Risk", "Etkisi", "Karar"],
        ["Breizh nadir sınıfları", "Macro‑F1 yüksek varyanslı olur", "Destek ve güven aralığı zorunlu; önceden tanımlı duyarlılık analizi"],
        ["TUM test desteği", "Şeker pancarı ölçülemez; patates tek örnek", "İkincil stres testi; ana sonuç yapılmaz"],
        ["Koordinat yokluğu", "10 km blok iddiası uygulanamaz", "Geometri edin ya da iddiayı kaldır"],
        ["Veri setleri farklı", "Ham skor karşılaştırması anlamsız", "Her veri setinde ayrı deney ve yorum"],
        ["Hesap yükü", "608 bin parsel tüm modellerde pahalı", "Disk destekli okuma, float32, önce %10 pilot; test küçültülmez"],
    ]
    story += [table(risk_rows, [3.6 * cm, 6.1 * cm, 7.2 * cm])]
    story += [callout(
        "Sonuç",
        "Faz 1'in veri inceleme bölümü somuttur ve gerçek dosyalara dayanır. Projenin zayıf noktası model seçimi değil, mekânsal bölme verisinin eksikliği ve TUM test sınıf desteğidir. Bu iki sorun çözülmeden 'coğrafi genelleme başarıyla ölçüldü' sonucu yazılamaz.",
        warning=True,
    )]
    story += [PageBreak()]

    # Appendix and references
    story += [h1("Ek A. Dosya bütünlüğü ve tekrar üretilebilirlik")]
    hash_rows = [
        ["Dosya", "Boyut (bayt)", "SHA‑256"],
        ["classmapping.csv", "441", "E850F71D…FAEC42E9"],
        ["frh01.csv", "13.916.067", "72E25583…49552A4"],
        ["frh02.csv", "11.158.458", "679F752D…44B26DB8"],
        ["frh03.csv", "12.616.127", "D8D0C60D…56838732"],
        ["frh04.csv", "9.628.839", "3EBAFE5F…DB03C789"],
        ["data2016-2018.xlsx", "5.785.624", "EB9EA9B2…83CA4C85C"],
        ["TestData.xlsx", "991.429", "8B742B62…39B40825"],
    ]
    story += [table(hash_rows, [5.1 * cm, 3.3 * cm, 8.5 * cm], alignments=["L", "C", "C"])]
    story += [p(
        "Tam özet değerleri yerel analiz çıktısında tutulmuştur. Büyük ham parsel dosyaları Git'e eklenmemeli; veri indirme adresleri ve SHA‑256 değerleri README içinde belgelenmelidir."
    )]
    story += [h1("Kaynakça")]
    refs = [
        "[1] Rußwurm, M. vd. (2020). <i>BreizhCrops: A Time Series Dataset for Crop Type Mapping.</i> ISPRS Archives, XLIII‑B2, 1545–1551. <link href='https://doi.org/10.5194/isprs-archives-XLIII-B2-2020-1545-2020'>https://doi.org/10.5194/isprs-archives-XLIII-B2-2020-1545-2020</link>",
        "[2] Rußwurm, M. ve Körner, M. (2020). <i>Self‑Attention for Raw Optical Satellite Time Series Classification.</i> ISPRS Journal of Photogrammetry and Remote Sensing, 169, 421–435. <link href='https://doi.org/10.1016/j.isprsjprs.2020.06.006'>https://doi.org/10.1016/j.isprsjprs.2020.06.006</link>",
        "[3] Martini, M. vd. (2021). <i>Domain‑Adversarial Training of Self‑Attention‑Based Networks for Land Cover Classification Using Multi‑Temporal Sentinel‑2 Satellite Imagery.</i> Remote Sensing, 13(13), 2564. <link href='https://doi.org/10.3390/rs13132564'>https://doi.org/10.3390/rs13132564</link>",
        "[4] Rußwurm, M. vd. (2023). <i>End‑to‑End Learned Early Classification of Time Series for In‑Season Crop Type Mapping.</i> ISPRS Journal of Photogrammetry and Remote Sensing, 196, 445–456. <link href='https://doi.org/10.1016/j.isprsjprs.2022.12.016'>https://doi.org/10.1016/j.isprsjprs.2022.12.016</link>",
        "[5] Yuan, X. vd. (2023). <i>Feature Importance Ranking of Random Forest‑Based End‑to‑End Learning Algorithm.</i> Remote Sensing, 15(21), 5203. <link href='https://doi.org/10.3390/rs15215203'>https://doi.org/10.3390/rs15215203</link>",
        "[6] Antonijević, O. vd. (2023). <i>Transfer Learning Approach Based on Satellite Image Time Series for the Crop Classification Problem.</i> Journal of Big Data, 10, 54. <link href='https://doi.org/10.1186/s40537-023-00735-2'>https://doi.org/10.1186/s40537-023-00735-2</link>",
        "[7] Marszalek, M., Lösch, M., Körner, M. ve Schmidhalter, U. (2022). <i>Early Crop‑Type Mapping Under Climate Anomalies.</i> Preprints.org, hakem değerlendirmesinden geçmemiş ön baskı. <link href='https://doi.org/10.20944/preprints202004.0316.v2'>https://doi.org/10.20944/preprints202004.0316.v2</link>",
        "[8] BreizhCrops resmî kod/veri deposu. <link href='https://github.com/dl4sits/BreizhCrops'>https://github.com/dl4sits/BreizhCrops</link> (erişim: 03.10.2026).",
        "[9] TUM‑CPN CropTypes resmî veri ve örnek kod deposu. <link href='https://github.com/TUM-CPN/CropTypes'>https://github.com/TUM-CPN/CropTypes</link> (erişim: 03.10.2026).",
        "[10] Uygulamalı Yapay Zeka Dersi Dönem Projesi Tebligatı, Faz 1 rapor maddeleri (ders içi belge, 2026).",
    ]
    for ref in refs:
        story += [p(ref, "Ref")]

    doc.build(story)
    print(PDF_PATH)


if __name__ == "__main__":
    build()
