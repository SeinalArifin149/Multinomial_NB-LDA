import pandas as pd
import re
import emoji
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 1. LOAD DATA
df = pd.read_csv('TikTok_Review_400000_2026.csv')

# --- BAGIAN PREPROCESSING (PENGGANTI GENSIM) ---
nama_kolom_teks = 'content'

print("Sedang memproses teks...")
# Fungsi pengganti gensim.utils.simple_preprocess
def clean_and_tokenize(text):
    text = str(text).lower()
    # deteksi emoji
    text = emoji.demojize(text, language='id')
    # Hapus karakter selain huruf & angka
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Ambil kata yang panjangnya minimal 2 huruf (sama seperti default gensim)
    words = [w for w in text.split() if len(w) >= 2]
    return " ".join(words)

# Terapkan ke dataframe
docs = df[nama_kolom_teks].apply(clean_and_tokenize).tolist()

# Membuat Dictionary & Corpus (Digantikan oleh CountVectorizer)
vectorizer = CountVectorizer()
tf = vectorizer.fit_transform(docs)
# ------------------------------------------

# 2. TRAIN MODEL LDA
print("Sedang melatih model LDA...")
num_topics = 4

lda_model = LatentDirichletAllocation(
    n_components=num_topics,
    random_state=42,
    max_iter=20,          # Ekivalen dengan 'passes' di gensim
    learning_method='batch',
    evaluate_every=-1
)
lda_model.fit(tf)

# 3. CETAK TOPIK (FORMAT GENSIM)
print("\nHasil Topik:")
feature_names = vectorizer.get_feature_names_out()
num_words = 10

for topic_idx, topic in enumerate(lda_model.components_):
    # Normalisasi bobot menjadi probabilitas agar formatnya seperti Gensim
    topic_probs = topic / topic.sum()
    
    # Ambil indeks 10 kata dengan probabilitas tertinggi
    top_features_ind = topic_probs.argsort()[:-num_words - 1:-1]
    
    # Gabungkan dalam format: probabilitas*"kata"
    topic_str_parts = []
    for i in top_features_ind:
        prob = topic_probs[i]
        word = feature_names[i]
        topic_str_parts.append(f'{prob:.3f}*"{word}"')
        
    topic_string = " + ".join(topic_str_parts)
    print(f"Topik #{topic_idx+1}: {topic_string}")

# 4. DISTRIBUSI TOPIK PER DOKUMEN
# Mengambil probabilitas topik per dokumen
doc_topic_dist = lda_model.transform(tf)

# Simpan ke dataframe
topic_data = []
for doc_probs in doc_topic_dist:
    doc_dict = {}
    for t, prob in enumerate(doc_probs):
        doc_dict[f"topic_{t}"] = prob
    topic_data.append(doc_dict)

topic_df = pd.DataFrame(topic_data)
topic_df = topic_df.fillna(0) # Isi 0 jika dokumen tidak mengandung topik tersebut

# 5. GABUNGKAN & SIMPAN
df_lda = pd.concat([df, topic_df], axis=1)

# Simpan Hasil
df_lda.to_csv("hasil_lda_4_topics.csv", index=False)

# Di scikit-learn, model disimpan menggunakan joblib. 
# Kita juga harus menyimpan vectorizer-nya untuk prediksi data baru nanti.
joblib.dump(lda_model, "lda_model_4_topics.joblib")
joblib.dump(vectorizer, "vectorizer_lda.joblib")

print("\n" + "="*30)
print("✓ Model LDA (Scikit-Learn) selesai dilatih!")
print("✓ Distribusi topik tersimpan di variabel 'df_lda'")
print("✓ File 'hasil_lda_4_topics.csv' telah dibuat.")
print("✓ Model tersimpan: 'lda_model_4_topics.joblib'")
print("="*30)