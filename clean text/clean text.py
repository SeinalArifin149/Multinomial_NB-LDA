import re
import emoji
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm import tqdm

# ==================================================
# LOAD DATA
# ==================================================

print("Membaca dataset...")

df = pd.read_csv("hasil_akhir_sentiment_per_topic.csv")

# TEST DULU 1000 DATA
# HAPUS BARIS INI SAAT SUDAH YAKIN
df = df.head(1000)

# ==================================================
# LOAD KAMUS SLANG
# ==================================================

print("Membaca kamus slang...")

slang_df = pd.read_csv("slang.csv")

slang_dict = dict(
    zip(
        slang_df["slang"].astype(str).str.lower(),
        slang_df["formal"].astype(str).str.lower()
    )
)

# ==================================================
# STEMMER
# ==================================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ==================================================
# STOPWORDS
# ==================================================

stop_words = set(stopwords.words("indonesian"))

custom_stop_words = {
    'yg', 'dg', 'rt', 'dgn', 'ny',
    'klo', 'amp', 'biar', 'bikin',
    'bilang', 'nih', 'sih',
    'tau', 'tuh', 'utk',
    'jd', 'jgn', 'sdh',
    'hehe', 'pen', 'nan',
    'loh', 'pas', 'buat'
}

stop_words.update(custom_stop_words)

# ==================================================
# NORMALISASI HURUF BERULANG
# ==================================================

def normalize_repeated(word):
    return re.sub(r'(.)\1{2,}', r'\1', word)

# ==================================================
# PREPROCESSING
# ==================================================

def preprocess_text(text):

    # Antisipasi nilai kosong
    if pd.isna(text):
        return ""

    text = str(text)

    # ==================================================
    # 1. Case Folding
    # ==================================================
    text = text.lower()

    # ==================================================
    # 2. Hapus URL
    # ==================================================
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # ==================================================
    # 3. Hapus Emoji
    # ==================================================
    text = emoji.replace_emoji(text, replace='')

    # ==================================================
    # 4. Hapus Mention (@)
    # ==================================================
    text = re.sub(r'@\w+', '', text)

    # ==================================================
    # 5. Hapus Hashtag (#)
    # ==================================================
    text = re.sub(r'#\w+', '', text)

    # ==================================================
    # 6. Hapus Angka
    # ==================================================
    text = re.sub(r'\d+', '', text)

    # ==================================================
    # 7. Hapus Tanda Baca
    # ==================================================
    text = re.sub(r'[^\w\s]', ' ', text)

    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # ==================================================
    # 8. Tokenizing
    # ==================================================
    tokens = text.split()

    # ==================================================
    # 9. Normalisasi Huruf Berulang
    # ==================================================
    tokens = [
        normalize_repeated(word)
        for word in tokens
    ]

    # ==================================================
    # 10. Normalisasi Slang
    # ==================================================
    tokens = [
        slang_dict.get(word, word)
        for word in tokens
    ]

    # ==================================================
    # 11. Stopword Removal
    # ==================================================
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # ==================================================
    # 12. Stemming
    # ==================================================
    text = " ".join(tokens)
    text = stemmer.stem(text)

    tokens = text.split()

    # ==================================================
    # 13. Hapus Kata Pendek
    # ==================================================
    tokens = [
        word
        for word in tokens
        if len(word) > 2
    ]

    # ==================================================
    # 14. Join Kembali
    # ==================================================
    return " ".join(tokens)

# ==================================================
# PROSES DATA
# ==================================================

print("Memulai preprocessing...")

tqdm.pandas()

df["clean_text"] = df["content"].progress_apply(
    preprocess_text
)

# Hapus hasil kosong
df = df[
    df["clean_text"].str.strip() != ""
]

# ==================================================
# SIMPAN
# ==================================================

output_file = "Hasil_Akhir.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

# ==================================================
# HASIL
# ==================================================

print("\n===================================")
print("PREPROCESSING SELESAI")
print("===================================")
print(f"File tersimpan : {output_file}")
print(f"Jumlah data    : {len(df)}")

print("\nContoh hasil:")
print(df[["content", "clean_text"]].head())