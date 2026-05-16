"""
generate_report_pdf.py — Generates a PDF report of the RNA DRL project.

Usage:
    pip install fpdf2
    python generate_report_pdf.py
"""
from fpdf import FPDF
import os


class ReportPDF(FPDF):
    @staticmethod
    def tr_ascii(text):
        """Convert Turkish special characters to ASCII equivalents."""
        tr_map = str.maketrans(
            "ıİşŞçÇğĞöÖüÜ",
            "iIsScCgGoOuU"
        )
        return text.translate(tr_map)

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "RNA Inverse Folding - Multi-Objective DRL Pipeline Report", align="C")
        self.ln(4)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Sayfa {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 60, 120)
        self.cell(0, 10, self.tr_ascii(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def sub_title(self, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 7, self.tr_ascii(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, self.tr_ascii(text))
        self.ln(1)

    def bold_text(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, self.tr_ascii(text))
        self.ln(1)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 245)
        self.set_text_color(50, 50, 50)
        x = self.get_x()
        self.set_x(x + 5)
        for line in text.split("\n"):
            self.cell(180, 5, self.tr_ascii(line), fill=True, new_x="LMARGIN", new_y="NEXT")
            self.set_x(x + 5)
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, self.tr_ascii(h), border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 0:
                self.set_fill_color(245, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, self.tr_ascii(str(cell)), border=1, fill=True, align="C")
            self.ln()
        self.ln(2)

    def note_box(self, text, box_type="NOTE"):
        colors = {
            "NOTE": (220, 235, 255),
            "WARNING": (255, 243, 220),
            "IMPORTANT": (255, 225, 225),
        }
        bg = colors.get(box_type, (240, 240, 240))
        self.set_fill_color(*bg)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(60, 60, 60)
        self.cell(190, 6, f"  {box_type}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.multi_cell(190, 5, self.tr_ascii(f"  {text}"), fill=True)
        self.ln(2)


def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ==================== COVER ====================
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 15, "RNA Inverse Folding", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Multi-Objective Deep Reinforcement Learning", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, "Pipeline Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Proje Durumu: Baseline Egitim Tamamlandi", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Tarih: 2026-05-04", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_draw_color(30, 60, 120)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Araclar: Gymnasium, Stable-Baselines3, ViennaRNA, TensorBoard", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Algoritmalar: PPO, DQN | Veri Seti: Eterna100-V2", align="C", new_x="LMARGIN", new_y="NEXT")

    # ==================== 1. PROJE AMACI ====================
    pdf.add_page()
    pdf.section_title("1. Proje Amaci ve Kapsami")
    pdf.body_text(
        "Bu proje, RNA inverse folding problemini cozmek icin cok amacli derin pekistirmeli "
        "ogrenme (Multi-Objective DRL) pipeline'i gelistirmeyi amaclamaktadir. Ajan, verilen bir "
        "hedef ikincil yapiyi (dot-bracket notasyonu) olusturacak bir RNA nukleotid dizisi tasarlar."
    )

    pdf.sub_title("1.1 Optimize Edilen 4 Amac")
    pdf.add_table(
        ["Amac", "Aciklama", "Aralik"],
        [
            ["R_struct", "Yapisal dogruluk (Hamming mesafesi)", "[0, 1]"],
            ["R_GC", "GC-icerigi (ideal %40-60 arasi)", "[0, 1]"],
            ["P_homo", "Homopolimer cezasi (4+ tekrar)", "[0, +)"],
            ["R_MFE", "Termodinamik kararlilik", "[0, +)"],
        ],
        [30, 100, 60],
    )

    pdf.bold_text("Bilesik odul: R = alpha * R_struct + beta * R_GC - gamma * P_homo + delta * R_MFE")

    # ==================== 2. ORTAM KURULUMU ====================
    pdf.section_title("2. Ortam Kurulumu")
    pdf.sub_title("2.1 Donanim")
    pdf.body_text(
        "Sistem: Intel CPU + NVIDIA RTX 3050 Ti. RL egitimi CPU-yogun oldugu icin "
        "yerel WSL2 ortami tercih edildi (Google Colab yerine)."
    )

    pdf.sub_title("2.2 Yazilim Ortami (WSL2 + Conda)")
    pdf.code_block(
        "conda create -n rlrna python=3.10 -y\n"
        "conda activate rlrna\n"
        "conda install -c bioconda -c conda-forge viennarna -y\n"
        "conda install -c conda-forge gsl -y\n"
        "pip install gymnasium stable-baselines3 tensorboard torch"
    )

    pdf.note_box(
        "ViennaRNA kurulumu: pip ile C derleme hatalari olusur, conda install -c bioconda zorunludur. "
        "Ayrica GSL kutuphanesi (libgsl.so.25) ayrica kurulmalidir.",
        "IMPORTANT"
    )

    # ==================== 3. GYMNASIUM ORTAMI ====================
    pdf.section_title("3. Gymnasium Ortami Tasarimi (environment.py)")

    pdf.sub_title("3.1 Episode Akisi")
    pdf.body_text(
        "1) reset(): Bos dizi, sifir observation.\n"
        "2) step(action): Her adimda bir nukleotid (A=0, C=1, G=2, U=3) eklenir.\n"
        "3) n adim sonunda dizi ViennaRNA ile katlanir ve 4 amac hesaplanir."
    )

    pdf.sub_title("3.2 Observation Space Evrimi")
    pdf.bold_text("Ilk Versiyon (Hatali - Ordinal Encoding):")
    pdf.body_text(
        "Her pozisyona 0-4 arasi sayi ataniyordu (0=bos, A=1, C=2, G=3, U=4). "
        "MLP agi 'G=3 > C=2' gibi sahte buyukluk iliskileri ogreniyordu."
    )
    pdf.bold_text("Duzeltilmis Versiyon (One-Hot Encoding):")
    pdf.body_text(
        "Her pozisyon 4-dim one-hot vektoru ile kodlandi: A=[1,0,0,0], C=[0,1,0,0], "
        "G=[0,0,1,0], U=[0,0,0,1], bos=[0,0,0,0]. "
        "Hedef yapi: 3-dim one-hot ('.'=[1,0,0], '('=[0,1,0], ')'=[0,0,1]). "
        "Toplam observation boyutu: 7n + 1."
    )

    pdf.sub_title("3.3 Potansiyel Tabanli Odul Sekillendirme (Ng et al. 1999)")
    pdf.body_text(
        "RNA inverse folding'de odul sadece episode sonunda verilir (seyrek odul). "
        "Cozum: Watson-Crick/wobble baz cifti eslesmelerine dayali potansiyel fonksiyonu."
    )
    pdf.code_block(
        "Phi(s) = dogru_baz_ciftleri / kontrol_edilen_cift_sayisi\n"
        "F(s, a, s') = 0.1 * (gamma_discount * Phi(s') - Phi(s))\n"
        "\n"
        "Garanti: Potansiyel tabanli sekillendirme optimal politikayi degistirmez."
    )

    # ==================== 4. VERI SETI ====================
    pdf.add_page()
    pdf.section_title("4. Veri Seti: Eterna100-V2")
    pdf.body_text(
        "Eterna100 benchmark'indan 20 hedef secildi: 15 egitim, 5 test. "
        "Hedefler farkli zorluk seviyeleri ve uzunluklari kapsayacak sekilde secildi."
    )

    pdf.sub_title("4.1 Egitim Seti (15 hedef)")
    pdf.add_table(
        ["#", "Puzzle", "Uzunluk", "Yapi Tipi"],
        [
            ["1", "Simple Hairpin", "18", "Basit hairpin"],
            ["8", "G-C Placement", "12", "Basit stem-loop"],
            ["10", "Frog Foot", "45", "Multi-stem"],
            ["13", "square", "67", "Ic ice stem"],
            ["15", "Small and Easy 6", "30", "Bulge + ic loop"],
            ["23", "Shortie 4", "17", "Cift hairpin"],
            ["25", "The Ministry", "62", "Ic ice hairpin"],
            ["26", "stickshift", "26", "Bulge yapisi"],
            ["30", "Corner bulge", "31", "Bulge training"],
            ["40", "Tripod5", "38", "Uclu dal"],
            ["41", "Shortie 6", "35", "Dortlu hairpin"],
            ["45", "5-Stack Branch", "63", "Coklu dal"],
            ["47", "Misfolded Aptamer", "50", "Yanlis katlanma"],
            ["54", "7 multiloop", "92", "7'li multiloop"],
            ["65", "Branching Loop", "40", "Dallanma"],
        ],
        [12, 65, 25, 88],
    )

    # ==================== 5. ADAPTIF AGIRLIK ====================
    pdf.section_title("5. Adaptif Agirlik Cizelgeleme (Curriculum Learning)")
    pdf.body_text(
        "Uc fazli gecis stratejisi kullanildi:\n\n"
        "Phase A (ilk %30): Sadece yapisal odul (alpha=1.0, geri kalan=0). "
        "Ajan once hedef yapiyi ogrenir.\n\n"
        "Phase B (%30-70): Diger amaclar lineer olarak rampa yapar. "
        "Alpha yavasce hedef degerine duser.\n\n"
        "Phase C (son %30): Sabit agirliklarla cok amacli optimizasyon."
    )

    pdf.sub_title("5.1 Grid Search Kombinasyonlari")
    pdf.add_table(
        ["Config", "alpha", "beta", "gamma", "delta", "Strateji"],
        [
            ["0", "0.5", "0.2", "0.1", "0.2", "Dengeli"],
            ["1", "0.6", "0.15", "0.1", "0.15", "Yapi agirlikli"],
            ["2", "0.4", "0.2", "0.15", "0.25", "MFE agirlikli"],
        ],
        [20, 25, 25, 25, 25, 70],
    )

    # ==================== 6. EGITIM ====================
    pdf.section_title("6. Egitim Pipeline'i")

    pdf.sub_title("6.1 Algoritma Karsilastirmasi")
    pdf.add_table(
        ["Ozellik", "PPO", "DQN"],
        [
            ["Policy", "On-policy", "Off-policy"],
            ["Replay Buffer", "Yok", "10K"],
            ["Exploration", "Entropy bonus", "e-greedy (1.0->0.05)"],
            ["Avantaj", "Kararli egitim", "Seyrek odulde iyi"],
        ],
        [50, 70, 70],
    )

    pdf.sub_title("6.2 Adaptif Timestep Olceklendirme")
    pdf.body_text(
        "Ilk deneylerde sabit 50K timestep kullanildiginda kisa puzzle'lar cozulurken "
        "uzun olanlar yetersiz episode aliyordu. Cozum: timesteps = seq_len x min_episodes."
    )
    pdf.add_table(
        ["Puzzle", "Uzunluk", "Sabit 50K (episode)", "Adaptif (episode)"],
        [
            ["P8", "12 nt", "4,166", "3,000"],
            ["P10", "45 nt", "1,111", "3,000"],
            ["P54", "88 nt", "568", "3,000"],
        ],
        [30, 40, 60, 60],
    )

    # ==================== 7. SONUCLAR ====================
    pdf.add_page()
    pdf.section_title("7. Deneysel Sonuclar")

    pdf.sub_title("7.1 Uc Run'in Karsilastirmasi (PPO, Config 0, Seed 42)")
    pdf.add_table(
        ["Puzzle", "Len", "Run1(50K)", "Run2(50K+fix)", "Run3(adapt)", "Durum"],
        [
            ["P1 Hairpin", "18", "0.993", "0.980", "0.998", "Cozuldu"],
            ["P8 G-C", "12", "0.997", "0.990", "0.980", "Cozuldu"],
            ["P26 stick", "26", "0.918", "0.846", "0.912", "Yaklasti"],
            ["P30 bulge", "31", "0.747", "0.858", "0.932", "Yaklasti"],
            ["P15 Easy6", "30", "0.525", "0.533", "0.864", "Yaklasti"],
            ["P10 Frog", "45", "0.556", "0.502", "0.747", "Yaklasti"],
            ["P65 Branch", "40", "0.562", "0.691", "0.691", "Kismi"],
            ["P13 square", "67", "0.718", "0.757", "0.748", "Kismi"],
            ["P40 Tripod", "38", "0.576", "0.592", "0.618", "Kismi"],
            ["P47 Aptamer", "50", "0.560", "0.560", "0.560", "Yetersiz"],
            ["P41 Short6", "35", "0.543", "0.543", "0.543", "Yetersiz"],
            ["P23 Short4", "17", "0.529", "0.529", "0.529", "Yetersiz"],
            ["P25 Minist", "62", "0.516", "0.516", "0.516", "Yetersiz"],
            ["P54 7multi", "92", "0.357", "0.396", "0.397", "Yetersiz"],
            ["P45 5stack", "63", "0.446", "0.385", "0.367", "Yetersiz"],
        ],
        [28, 12, 28, 32, 28, 24],
    )

    pdf.sub_title("7.2 Ozet Istatistikleri")
    pdf.add_table(
        ["Metrik", "Run 1", "Run 2", "Run 3"],
        [
            ["Cozulen (>50% success)", "2/15", "2/15", "2/15"],
            ["Yaklasan (best >= 0.8)", "4/15", "4/15", "6/15"],
            ["Ortalama R_struct", "0.640", "0.643", "0.690"],
        ],
        [60, 40, 40, 40],
    )

    pdf.sub_title("7.3 Bulgular")
    pdf.bold_text("Basarilar:")
    pdf.body_text(
        "- Pipeline uctan uca calisiyor: ortam -> egitim -> loglama -> analiz\n"
        "- Kisa yapilar (%94-98 basari) guvenilir sekilde cozuluyor\n"
        "- Adaptif timestep ile 4 puzzle daha yaklasma kategorisine yukseldi\n"
        "- Ng et al. potansiyel tabanli sekillendirme yogun sinyal sagliyor\n"
        "- Adaptif agirlik cizelgeleme (Phase A->B->C) duzgun calisiyor"
    )
    pdf.bold_text("Sinirliliklar:")
    pdf.body_text(
        "- MLP mimarisi uzun dizilerde (>40nt) yetersiz kaliyor\n"
        "- Arama uzayi 4^n ile patliyor (92nt icin ~10^55 olasi dizi)\n"
        "- Bazi multiloop yapilari kisa olmalar\u0131na ragmen cozulemiyor\n"
        "- 3000 episode bazi karmasik yapilar icin yetersiz"
    )

    # ==================== 8. COZULEN SORUNLAR ====================
    pdf.add_page()
    pdf.section_title("8. Cozulen Teknik Sorunlar")

    pdf.sub_title("8.1 ViennaRNA Kurulum Zinciri")
    pdf.code_block(
        "Problem: import RNA -> ModuleNotFoundError\n"
        "Cozum:   conda install -c bioconda viennarna\n"
        "\n"
        "Problem: libgsl.so.25 not found\n"
        "Cozum:   conda install -c conda-forge gsl\n"
        "\n"
        "Problem: Yanlis Python calisiyor (3.13 vs 3.10)\n"
        "Cozum:   Tam yol: /home/.../envs/rlrna/bin/python"
    )

    pdf.sub_title("8.2 Sabit R_struct (0.3333) Sorunu")
    pdf.body_text(
        "Belirti: r_struct 50K step boyunca 0.3333'te sabit kaldi.\n"
        "Teshis: ViennaRNA kurulu degildi, mock_fold() her zaman '...' donduruyordu. "
        "1 - 8/12 = 0.3333 (hedef yapidaki parantez sayisi).\n"
        "Cozum: ViennaRNA + GSL kurulumu."
    )

    pdf.sub_title("8.3 Ordinal Encoding -> One-Hot Fix")
    pdf.body_text(
        "Belirti: Ajan rastgele seviyenin ustune cikamiyordu.\n"
        "Teshis: A=1, C=2, G=3, U=4 ordinal kodlama sahte buyukluk iliskisi yaratiyordu.\n"
        "Cozum: 4-dim one-hot encoding uyguland\u0131."
    )

    pdf.sub_title("8.4 Shaping Reward Dominasyonu")
    pdf.body_text(
        "Belirti: Ajan terminal odulu optimize etmek yerine ara odulleri takip ediyordu.\n"
        "Cozum: shaping_scale = 0.1 ile ara oduller kucultuldu."
    )

    # ==================== 9. DOSYA YAPISI ====================
    pdf.section_title("9. Proje Dosya Yapisi")
    pdf.code_block(
        "d:\\RL_project\\\n"
        "+-- environment.py          Gymnasium ortami (4 obj + Ng shaping)\n"
        "+-- eterna100.py            Eterna100-V2 veri seti (15+5)\n"
        "+-- baseline.py             PPO Phase A baseline\n"
        "+-- baseline_dqn.py         DQN Phase A baseline\n"
        "+-- train_multi_target.py   Tam cok-amacli egitim pipeline\n"
        "+-- analyze_results.py      Sonuc analiz scripti\n"
        "+-- preflight_check.py      On-ucus kontrol scripti\n"
        "+-- models/                 Egitilmis modeller (15 adet)\n"
        "+-- tensorboard_logs/       TensorBoard loglari"
    )

    # ==================== 10. SONRAKI ADIMLAR ====================
    pdf.section_title("10. Sonraki Adimlar")
    pdf.body_text(
        "1. DQN karsilastirmasi: Ayni hedeflerle DQN egitimi calistirma\n"
        "2. Grid Search: 3 weight config x 3 seed x 2 algo = 18 deney\n"
        "3. Test seti degerlendirmesi: 5 test hedefinde model performansi\n"
        "4. Ablasyon calismasi: Her odul bilesenini tek tek kaldirma\n"
        "5. Pareto analizi: R_struct vs R_GC tradeoff grafikleri"
    )

    return pdf


if __name__ == "__main__":
    pdf = build_report()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_report.pdf")
    pdf.output(output_path)
    print(f"PDF rapor olusturuldu: {output_path}")
    print(f"Boyut: {os.path.getsize(output_path) / 1024:.1f} KB")
