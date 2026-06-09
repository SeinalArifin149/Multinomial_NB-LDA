import pandas as pd
import os

# 1. Deteksi otomatis folder
lokasi_skrip = os.path.dirname(os.path.abspath(__file__))

PATH_DATA_ASLI = os.path.join(lokasi_skrip, 'TikTok_Review_400000_2026.csv')
PATH_DATA_PREP = os.path.join(lokasi_skrip, 'hasil_preprocessing.csv')
PATH_HASIL_AKHIR = os.path.join(lokasi_skrip, 'data_untuk_teman.csv')

print("Memulai proses penggabungan data...")

try:
    # 2. Baca file
    df_asli = pd.read_csv(PATH_DATA_ASLI)
    df_prep = pd.read_csv(PATH_DATA_PREP)
    
    # --- PERBAIKAN DI SINI ---
    # 3. Hapus baris duplikat di data prep agar RAM tidak meledak saat di-merge
    df_prep_unik = df_prep.drop_duplicates(subset=['content'])
    
    # 4. Gabungkan (Merge) secara aman
    df_gabungan = pd.merge(df_asli, df_prep_unik, on='content', how='left')
    
    # 5. Ganti isi kolom 'content'
    df_gabungan['content'] = df_gabungan['final_clean_text']
    df_gabungan = df_gabungan.drop(columns=['final_clean_text'])
    
    # 6. Bersihkan baris kosong (jika ada yang tidak mendapatkan pasangan)
    df_gabungan['content'] = df_gabungan['content'].fillna('')
    
    # 7. Simpan file
    df_gabungan.to_csv(PATH_HASIL_AKHIR, index=False)
    
    print(f"BERHASIL! Data tidak meledak dan siap dikirim.")
    print(f"File disimpan sebagai: {PATH_HASIL_AKHIR}")
    
except Exception as e:
    print(f"Terjadi kesalahan: {e}")