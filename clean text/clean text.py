import re
import emoji
import nltk
import pandas as pd
from nltk.corpus import stopwords

df = pd.read_csv("hasil_akhir_sentiment_per_topic.csv")
# slang = pd.read_csv("slang.csv")

def clean_word(text):
    if not isinstance (text,str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = emoji.replace_emoji(text,replace="")
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('indonesian'))

    custom_stop_words = {
            'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 'biar', 
            'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 
            'tau', 'tdk', 'tuh', 'utk', 'ya', 'jd', 'jgn', 'sdh', 'aja', 'n', 
            't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt', 'dan', 'di', 
            'ke', 'dari', 'ini', 'itu', 'lagi', 'kok', 'pas', 'kan', 'aku', 
            'kamu', 'dia', 'mereka', 'kita', 'buat', 'ada', 'udah'
        }
    stop_words.update(custom_stop_words)

    words = text.split()
    cleaned_word = [word for word in words if word not in stop_words and len(word) > 1]
    return " ".join(cleaned_word)


df["df_topic"] = df["content"].apply(clean_word)

df.to_csv("Hasil Akhir.csv",index=False, encoding='utf-8')