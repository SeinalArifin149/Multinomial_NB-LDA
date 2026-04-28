 code$ python train-LDA_with_scikit.py 
Sedang memproses teks...
Sedang melatih model LDA...

Hasil Topik:
Topik #1: 0.033*"saya" + 0.025*"di" + 0.024*"tiktok" + 0.024*"ke" + 0.022*"akun" + 0.019*"tidak" + 0.018*"ada" + 0.017*"bisa" + 0.015*"jempol" + 0.014*"yang"
Topik #2: 0.039*"di" + 0.030*"tiktok" + 0.027*"bisa" + 0.022*"nya" + 0.018*"lagi" + 0.016*"buka" + 0.013*"tolong" + 0.013*"ini" + 0.011*"harus" + 0.011*"sering"
Topik #3: 0.115*"bagus" + 0.055*"sangat" + 0.032*"aplikasi" + 0.029*"dan" + 0.026*"ini" + 0.022*"tiktok" + 0.019*"tangan" + 0.019*"banget" + 0.018*"suka" + 0.018*"mantap"
Topik #4: 0.145*"wajah" + 0.047*"dengan" + 0.037*"seru" + 0.033*"hati" + 0.029*"mata" + 0.029*"tersenyum" + 0.022*"ok" + 0.020*"sangat" + 0.019*"bintang" + 0.016*"marah"

==============================
✓ Model LDA (Scikit-Learn) selesai dilatih!
✓ Distribusi topik tersimpan di variabel 'df_lda'
✓ File 'hasil_lda_4_topics.csv' telah dibuat.
✓ Model tersimpan: 'lda_model_4_topics.joblib'
==============================
Topic||Kata kunci
Topic #1 = Masalah Seputar Akun & Akses (Cenderung Negatif)  {"saya", "akun", "tidak", "bisa", "jempol"}
Topik #2: Keluhan Teknis & Bug Aplikasi (Cenderung Negatif){"bisa", "buka", "tolong", "sering", "lagi"}
Topik #3: Review Sangat Positif & Apresiasi (Sentimen Positif) {"bagus", "sangat", "banget", "suka", "mantap", "tangan"}
Topik #4: Reaksi Emosional & Hiburan (Sentimen Positif) {Kata kunci: "wajah", "hati", "mata", "tersenyum", "seru", "bintang", "marah"}