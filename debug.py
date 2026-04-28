import pandas as pd

# LOAD DATA
df = pd.read_csv('TikTok_Review_400000_2026.csv')
# df = pd.read_csv('Hasil LDA.csv')

# =========================
# 1. DIMENSI DATA
# =========================
print("📊 DIMENSI DATA")
print(f"Rows  : {df.shape[0]}")
print(f"Colom : {df.shape[1]}")

# =========================
# 2. MISSING VALUE (NULL)
# =========================
print("\n❗ MISSING VALUE")
print(df.isnull().sum())

# =========================
# 3. DUPLICATED DATA
# =========================
print("\n🔁 DUPLICATED")
print(f"Jumlah duplikat: {df.duplicated().sum()}")