import os
import time
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==========================================
# KONFIGURASI
# ==========================================
PATH_DATASET    = "../prep siigit/hasil_preprocessing.csv"
KOLOM_TEKS      = "final_clean_text"  
KOLOM_SCORE     = "score"             # Kolom rating asli (1 sampai 5)
KOLOM_LABEL     = "label"             # Kolom baru hasil mapping (0 atau 1)
MODEL_NAME      = "indobenchmark/indobert-base-p2"
MAX_LENGTH      = 128                 
BATCH_SIZE      = 32                  # Aman untuk GPU bawaan Kaggle Pro

print(" Memeriksa Device...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Menggunakan Device: {device.upper()}")

# ==========================================
# 1. LOAD, MAPPING LABEL, & CLEANING DATA
# ==========================================
print("\nMembaca dataset...")
df = pd.read_csv(PATH_DATASET)

# Pastikan tidak ada nilai NaN di kolom teks maupun score sebelum diproses
df = df.dropna(subset=[KOLOM_TEKS, KOLOM_SCORE])

print("Melakukan mapping kolom score ke label biner (0 dan 1)...")
# Logika: Jika score >= 4 maka 1 (Positif), jika di bawah 4 (1, 2, 3) maka 0 (Negatif)
df[KOLOM_LABEL] = df[KOLOM_SCORE].apply(lambda x: 1 if x >= 4 else 0)

# Hapus baris yang teksnya kosong ("") akibat efek samping preprocessing
df[KOLOM_TEKS] = df[KOLOM_TEKS].astype(str).str.strip()
df = df[df[KOLOM_TEKS] != ""]

print(f"Total data setelah dibersihkan: {len(df)} baris.")
print(df[KOLOM_LABEL].value_counts()) # Menampilkan distribusi jumlah data Positif vs Negatif

# Split data (80% Train, 20% Validasi)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[KOLOM_LABEL])
print(f"Jumlah Data Train: {len(train_df)} | Data Val: {len(val_df)}")

# Konversi ke format Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[[KOLOM_TEKS, KOLOM_LABEL]])
val_dataset = Dataset.from_pandas(val_df[[KOLOM_TEKS, KOLOM_LABEL]])

# ==========================================
# 2. TOKENISASI MULTI-PROCESSING
# ==========================================
print("\nMemuat Tokenizer IndoBERT...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    result = tokenizer(examples[KOLOM_TEKS], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    result["labels"] = examples[KOLOM_LABEL]
    return result

print("Menjalankan Tokenisasi (Menggunakan multi-processing CPU Kaggle)...")
train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=[KOLOM_TEKS, KOLOM_LABEL], num_proc=2)
val_tokenized = val_dataset.map(preprocess_function, batched=True, remove_columns=[KOLOM_TEKS, KOLOM_LABEL], num_proc=2)

# ==========================================
# 3. LOAD MODEL INDOBERT
# ==========================================
print("\nMemuat Model IndoBERT...")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ==========================================
# 4. TRAINING ARGUMENTS (OPTIMAL UNTUK KAGGLE)
# ==========================================
print("\nMenyusun parameter training...")
training_args = TrainingArguments(
    output_dir="./hasil_indobert_clean",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,              
    weight_decay=0.01,
    eval_strategy="epoch",        
    save_strategy="epoch",           
    load_best_model_at_end=True,     
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=500,               
    fp16=torch.cuda.is_available(),  # Mempercepat training menggunakan Tensor Cores GPU
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    compute_metrics=compute_metrics,
)

# ==========================================
# 5. EXECUTE TRAINING
# ==========================================
print("\n==================================================")
print(f"MEMULAI TRAINING INDOBERT DENGAN {len(df)} DATA...")
print("==================================================")
start_time = time.time()

trainer.train()

print(f"\nTraining Selesai dalam: {(time.time() - start_time)/60:.2f} Menit")

# ==========================================
# 6. SIMPAN HASIL MODEL
# ==========================================
output_model_dir = "./indobert_clean_sentiment_model"
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Model terbaik berhasil disimpan di: {output_model_dir}")