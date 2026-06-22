import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# import sys

# sys.stdout = open("hasil_lda.txt", "w", encoding="utf-8")

print("Membaca dataset...")

df = pd.read_csv("../prep siigit/hasil_preprocessing.csv")

docs = df["final_clean_text"].fillna("").astype(str)


print("Membuat Document-Term Matrix...")

vectorizer = CountVectorizer(
    min_df=20,
    max_df=0.8,
    ngram_range=(1, 2),
    max_features=10000
)

dtm = vectorizer.fit_transform(docs)


print("Melatih model LDA...")

NUM_TOPICS = 4

lda = LatentDirichletAllocation(
    n_components=NUM_TOPICS,
    random_state=42,
    max_iter=20,
    learning_method="batch"
)
import time

print("Mulai training LDA...")
start = time.time()

lda.fit(dtm)

print("Selesai dalam:", time.time() - start, "detik")


print("HASIL TOPIK")

feature_names = vectorizer.get_feature_names_out()

NUM_WORDS = 10

for topic_idx, topic in enumerate(lda.components_):

    top_indices = topic.argsort()[-NUM_WORDS:][::-1]

    top_words = [
        feature_names[i]
        for i in top_indices
    ]

    print(f"Topik {topic_idx + 1}")
    print(", ".join(top_words))
    print()


print("Menghitung distribusi topik...")

topic_distribution = lda.transform(dtm)

topic_df = pd.DataFrame(
    topic_distribution,
    columns=[
        f"topic_{i+1}"
        for i in range(NUM_TOPICS)
    ]
)

# Topik dominan tiap dokumen
topic_df["dominant_topic"] = (
    topic_df.idxmax(axis=1)
)

# ==========================================
# GABUNGKAN HASIL
# ==========================================

df_result = pd.concat(
    [df, topic_df],
    axis=1
)

# ==========================================
# SIMPAN HASIL
# ==========================================

output_file = "hasil_lda_4_topics.csv"

df_result.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

# =====================
print("\n==============================")
print("LDA SELESAI")
print("==============================")
print(f"Jumlah Data : {len(df)}")
print(f"Jumlah Topik: {NUM_TOPICS}")
print(f"Output      : {output_file}")