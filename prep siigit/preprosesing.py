import pandas as pd
import re
import os
import unicodedata
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# KONFIGURASI — SESUAIKAN DI SINI
# ==========================================

_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DATASET    = os.path.join(os.path.dirname(_DIR), 'TikTok_Review_400000_2026.csv')
KOLOM_TEKS      = 'content'
PATH_SLANG_CSV  = os.path.join(_DIR, 'slang.csv')       # kolom: slang, formal
PATH_WHITELIST  = os.path.join(_DIR, 'whitelist.csv')   # kolom: kata

# ==========================================
# TAHAP 0: BACA DATASET
# ==========================================

print("Membaca dataset...")
df = pd.read_csv(PATH_DATASET)
print("\nDataset Berhasil Dimuat!")
print(df.info())
print(df.head(5))

# ==========================================
# TAHAP 1: DATA CLEANING & CASE FOLDING
# ==========================================

import emoji
import unicodedata
import re

import emoji
import re
import unicodedata

import re

# ==========================================
# TAHAP 1: DATA CLEANING & CASE FOLDING (REVISI)
# ==========================================

# Kita tidak lagi memerlukan library emoji, daftar_emoji.txt, 
# atau fungsi hapus_semua_emoji yang berat.

def pembersihan_awal(text):
    # Memastikan input adalah teks (string)
    if not isinstance(text, str):
        return ""
        
    # Langkah 1: Lowercase (kecilkan huruf lebih dulu)
    text = text.lower()
    
    # Langkah 2: Hapus URL / Link website
    text = re.sub(r'https?://\s*\S+|www\.\s*\S+', ' ', text)
    
    # Langkah 3: Hapus Mention (@username)
    text = re.sub(r'@\s*\S+', ' ', text)
    
    # Langkah 4: SAPU BERSIH (Hapus Emoji, Angka, & Tanda Baca)
    # Menghapus semua karakter selain abjad a-z dan spasi.
    # Karakter yang dihapus diganti spasi (' ') agar kata tidak menempel.
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Langkah 5: Normalisasi spasi ganda yang tersisa akibat Langkah 4
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Menerapkan fungsi pembersihan ke dalam dataset
df['clean_step1'] = df[KOLOM_TEKS].apply(pembersihan_awal)

# Menampilkan hasil
print("\nHasil Pembersihan Tahap 1:")
print(df[[KOLOM_TEKS, 'clean_step1']].head(5))
# ==========================================
# TAHAP 2: PENANGANAN KARAKTER BERULANG
# ==========================================

def hapus_karakter_berulang(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'(.)\1+', r'\1\1', text)

df['clean_step2'] = df['clean_step1'].apply(hapus_karakter_berulang)
print("\nPerbandingan Tahap 1 & 2:")
print(df[['clean_step1', 'clean_step2']].head(5))

# ==========================================
# TAHAP 3: TOKENISASI
# ==========================================

def tokenisasi_teks(text):
    if not isinstance(text, str):
        return []
    return text.split()

df['clean_step3'] = df['clean_step2'].apply(tokenisasi_teks)
print("\nHasil Tokenisasi:")
print(df[['clean_step2', 'clean_step3']].head(5))

# ==========================================
# TAHAP 4: NORMALISASI TEKS
# ==========================================

# --- SUMBER 1: Kamus Publik (GitHub) ---
print("\n1. Mengunduh kamus publik dari GitHub...")
try:
    url_kamus = "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"
    df_publik = pd.read_csv(url_kamus)
    kamus_gabungan = dict(zip(df_publik['slang'], df_publik['formal']))
    print(f"   -> Berhasil memuat {len(kamus_gabungan)} kata dari kamus publik.")
except Exception as e:
    print(f"   -> Gagal mengunduh kamus publik: {e}")
    kamus_gabungan = {}

# --- SUMBER 2: slang.csv milikmu ---
print("2. Memuat kamus slang dari slang.csv...")
if os.path.exists(PATH_SLANG_CSV):
    try:
        # Gunakan sep=',' secara eksplisit, hapus engine='python'
        df_slang = pd.read_csv(
            PATH_SLANG_CSV, 
            sep=',', 
            encoding='utf-8-sig'
        )
        
        # Bersihkan spasi dan paksa jadi huruf kecil semua
        df_slang.columns = df_slang.columns.str.strip().str.lower()
        
        if 'slang' in df_slang.columns and 'formal' in df_slang.columns:

            df_slang = df_slang.dropna(subset=['slang', 'formal'])
            # Bersihkan spasi berlebih pada isi data
            kol_slang = df_slang['slang'].astype(str).str.strip().str.lower()
            kol_formal = df_slang['formal'].astype(str).str.strip().str.lower()
            
            kamus_slang = dict(zip(kol_slang, kol_formal))
            kamus_gabungan.update(kamus_slang)
            
            print(f"   -> BERHASIL! {len(kamus_slang)} kata dari slang.csv ditambahkan.")
        else:
            print(f"   -> ERROR KOLOM: Yang terbaca malah {list(df_slang.columns)}")
    except Exception as e:
        print(f"   -> ERROR BACA FILE: {e}")
else:
    print(f"   -> ERROR: File {PATH_SLANG_CSV} TIDAK DITEMUKAN di folder!")


# --- WHITELIST: whitelist.csv ---
print("3. Memuat whitelist dari whitelist.csv...")
whitelist_baku = set()
if os.path.exists(PATH_WHITELIST):
    try:
        df_wl = pd.read_csv(PATH_WHITELIST)
        df_wl.columns = df_wl.columns.str.strip()
        if 'kata' in df_wl.columns:
            whitelist_baku = set(df_wl['kata'].str.strip().str.lower())
            print(f"   -> Berhasil memuat {len(whitelist_baku)} kata baku dari whitelist.csv.")
        else:
            print(f"   -> ERROR: Kolom tidak sesuai! Kolom terbaca: {list(df_wl.columns)}")
            print("      Pastikan header baris pertama adalah 'kata'.")
    except Exception as e:
        print(f"   -> Gagal membaca whitelist.csv: {e}")
else:
    print(f"   -> File {PATH_WHITELIST} tidak ditemukan. Whitelist kosong.")
    print("      Buat file whitelist.csv dengan satu kolom 'kata' untuk menggunakannya.")

# --- FUNGSI NORMALISASI ---
def normalisasi(tokens):
    if not isinstance(tokens, list):
        return []
    hasil = []
    for kata in tokens:
        if kata in whitelist_baku:
            hasil.append(kata)  # kata baku, tidak diubah
        else:
            hasil.append(kamus_gabungan.get(kata, kata))
    return hasil

df['clean_step4'] = df['clean_step3'].apply(normalisasi)

print(f"\nTotal kata di kamus gabungan : {len(kamus_gabungan)}")
print(f"Total kata di whitelist      : {len(whitelist_baku)}")
print("\nHasil Normalisasi:")
print(df[['clean_step3', 'clean_step4']].head(5))

# ==========================================
# TAHAP 5: STOPWORD REMOVAL (REVISI)
# ==========================================

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

# Mengambil daftar stopword bawaan NLTK
daftar_stopword = set(stopwords.words('indonesian'))

# Daftar kata yang WAJIB DISELAMATKAN (tidak boleh dibuang)
# karena memiliki makna penting untuk klasifikasi sentimen
kata_penting_sentimen = {
    # Kata Negasi (menolak sentimen)
    'tidak', 'bukan', 'jangan', 'kurang', 'belum', 'enggak',
    
    # Kata Penguat (Intensifier sentimen)
    'sangat', 'banget', 'sekali', 'paling', 'terlalu', 'lumayan',
    
    # Kata Penghubung Kontras/Kondisi (mengubah arah sentimen)
    'tapi', 'padahal', 'terus', 'masa', 'kalau', 'karena',
    'walaupun', 'meskipun', 'namun'
}

# Mengecualikan kata-kata penting dari daftar stopword NLTK
stopword_aman = daftar_stopword - kata_penting_sentimen

def hapus_stopword(tokens):
    # Validasi input berupa list
    if not isinstance(tokens, list):
        return []
        
    # Menyaring kata: masukkan ke hasil HANYA JIKA kata 
    # tersebut tidak ada di dalam stopword_aman
    return [kata for kata in tokens if kata not in stopword_aman]

# Menerapkan fungsi ke kolom dataset
df['clean_step5'] = df['clean_step4'].apply(hapus_stopword)

print("\nHasil Stopword Removal (Revisi Sentimen):")
print(df[['clean_step4', 'clean_step5']].head(5))

# ==========================================
# TAHAP 6: STEMMING (nlp-id)
# ==========================================

print("\nMemuat Lemmatizer...")
try:
    from nlp_id.lemmatizer import Lemmatizer
    lemmatizer = Lemmatizer()

    def stemming_teks(tokens):
        if not isinstance(tokens, list) or len(tokens) == 0:
            return []
        kalimat_dasar = lemmatizer.lemmatize(" ".join(tokens))
        return kalimat_dasar.split()

    df['clean_step6'] = df['clean_step5'].apply(stemming_teks)
    print("\nHasil Stemming:")
    print(df[['clean_step5', 'clean_step6']].head(5))

except ImportError:
    print("PERINGATAN: library nlp-id tidak terinstall.")
    print("Install dengan: pip install nlp-id")
    print("Melewati tahap stemming...")
    df['clean_step6'] = df['clean_step5']

# ==========================================
# TAHAP 7: FINAL CLEAN TEXT & TF-IDF
# ==========================================

def gabung_token(tokens):
    return " ".join(tokens) if isinstance(tokens, list) else ""

df['final_clean_text'] = df['clean_step6'].apply(gabung_token)

print("\nMemproses TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['final_clean_text'])

print(f"\nUkuran Matriks TF-IDF: {tfidf_matrix.shape}")
print(f"  {tfidf_matrix.shape[0]} baris data, {tfidf_matrix.shape[1]} fitur kata")

daftar_kata = tfidf_vectorizer.get_feature_names_out()
df_tfidf_sample = pd.DataFrame(tfidf_matrix[:5].toarray(), columns=daftar_kata)
print("\nCuplikan Nilai TF-IDF:")
print(df_tfidf_sample.iloc[:, 10:15].head(5))

# ==========================================
# SIMPAN HASIL
# ==========================================

output_path = os.path.join(_DIR, 'hasil_preprocessing.csv')
df[[KOLOM_TEKS, 'final_clean_text']].to_csv(output_path, index=False)
print(f"\nHasil disimpan ke: {output_path}")
