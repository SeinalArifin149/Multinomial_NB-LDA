import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown # Library to download from GDrive

# ==========================================
# 1. PAGE CONFIGURATION & GLOBAL VARIABLES
# ==========================================
st.set_page_config(
    page_title="TikTok Sentiment Analysis App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. GOOGLE DRIVE DOWNLOAD FUNCTION
# ==========================================
@st.cache_resource
def download_files_from_drive():
    """
    Downloads files from Google Drive if not present locally.
    Uses @st.cache_resource to avoid re-downloading on rerun.
    """
    
    files = {
        "topik": {
            "id": "1DuFx9L1g__xBiZKFkTmIKQdBHRudq_DD",
            "output": "hasil_akhir_sentiment_per_topic_2026.csv"
        },
        "aspek": {
            "id": "1bn9ljnD9c__QIJ6oA1b-567j977X9_aO",
            "output": "tiktok_sentiment_analysis_results_2026.csv"
        }
    }
    
    paths = {}
    
    for key, info in files.items():
        output_path = info["output"]
        if not os.path.exists(output_path):
            with st.spinner(f"Downloading {key} data from Google Drive..."):
                url = f'https://drive.google.com/uc?id={info["id"]}'
                gdown.download(url, output_path, quiet=False)
        paths[key] = output_path
        
    return paths["topik"], paths["aspek"]

# --- EXECUTE DOWNLOAD ---
try:
    CSV_TOPIK_PATH, CSV_ASPEK_PATH = download_files_from_drive()
except Exception as e:
    st.error(f"Failed to download files from Google Drive: {e}")
    # Fallback filename to avoid total crash
    CSV_TOPIK_PATH = 'hasil_akhir_sentiment_per_topic.csv'
    CSV_ASPEK_PATH = 'tiktok_sentiment_analysis_results.csv'


# ==========================================
# 3. DICTIONARY UNTUK ASPEK DAN URUTAN TOPIK
# ==========================================
ASPEK_DICT = {
    "UI/UX": ["tampilan", "antarmuka", "desain", "tema", "warna", "navigasi", "ikon", "huruf", "gelap", "mudah", "transisi", "animasi", "responsif", "sederhana", "praktis", "scroll", "gerakan", "malam", "ikonografi", "struktur", "interaktif", "dasbor", "menu", "submenu", "tipografi", "keterbacaan", "akses", "pintasan", "seret", "cubitan", "perbesar", "sentuhan", "geser", "hambatan", "grid", "petunjuk", "widget", "slider", "tab", "panel", "breadcrumb", "tooltip", "popup", "tablet", "ponsel", "alur", "microinteraksi", "jelas", "kontras", "intuitif", "fokus", "efek", "keseragaman", "terang", "gelap", "aksesibilitas", "bersih", "simbol", "mudah dipahami", "fluid", "responsif", "tata layout", "geser seret", "sapuan", "sentuh", "pintasan", "layar penuh", "minimalis", "transparan", "sorot", "kontras", "padding", "margin", "ramah pengguna", "gesture sentuh", "drag & drop"],
    "Features": ["fitur", "fungsi", "pembaruan", "filter", "efek", "stiker", "emoji", "duet", "gabung", "siaran", "unduh", "unggah", "edit", "rekam", "musik", "suara", "daftar putar", "komen", "obrolan", "cerita", "sorotan", "templat", "tanda", "pin", "penanda", "multiakun", "sinkronisasi", "simpan", "otomatis", "rekomendasi", "draft", "pemberitahuan", "ulang", "bagikan", "cadangan", "pulihkan", "impor", "ekspor", "perpustakaan", "konten", "penyesuaian", "potong", "privat", "publik", "caption", "lokasi", "rekaman", "langsung", "transisi", "kecepatan", "kolaborasi", "animasi", "watermark", "mode gelap", "pintasan", "loop", "pin video", "favorit", "sorot", "simpan otomatis", "duet teman", "kolaborasi", "reaksi cerita", "musik cerita", "tag", "mention", "polling", "stiker cerita", "jadwal posting"],
    "Performance": ["cepat", "lambat", "lemot", "macet", "kesalahan", "memuat", "tutup", "terhenti", "bug", "beku", "tertunda", "fps", "optimal", "stabil", "respon", "memori", "prosesor", "grafik", "restart", "buffering", "panas", "waktu", "buruk", "ram", "baterai", "segarkan", "timeout", "drop", "perlambatan", "freeze", "lag", "loading", "macet klik", "panas berlebih", "cpu", "gpu", "refresh", "input", "server", "antarmuka", "pembaruan", "reaksi", "konten", "kecepatan", "lancar", "jitter", "lambat", "tunda", "crash", "hang", "restart aplikasi", "lag", "stutter", "startup", "shutdown", "boot", "fps drop", "gerakan lambat", "glitch"],
    "Security": ["privasi", "izin", "keamanan", "data", "akun", "blokir", "tangguh", "peretas", "penipuan", "kata sandi", "enkripsi", "otp", "phishing", "ilegal", "verifikasi", "pelacakan", "bocor", "login", "perlindungan", "autentikasi", "pemulihan", "captcha", "virus", "spam", "tautan", "pemantauan", "identitas", "sadap", "transaksi", "logout", "pengaturan", "sidik jari", "wajah", "lokasi", "sensitif", "aman", "pihak ketiga", "notifikasi", "peringatan", "firewall", "antivirus", "cadangan", "keamanan data", "token", "hash", "verifikasi otp", "aman", "enkripsi AES", "TLS", "multi-faktor", "authenticator", "perlindungan data", "intrusi", "malware", "perekam tombol", "kebocoran privasi", "peringatan keamanan"],
    "Service": ["layanan", "dukungan", "respon", "admin", "cs", "bantuan", "komplain", "laporan", "masukan", "obrolan", "tiket", "solusi", "panduan", "manual", "email", "tindak lanjut", "tutorial", "qa", "call center", "video", "teknis", "pengguna", "jawaban", "sopan", "lambat", "keluhan", "komunitas", "panduan lengkap", "dukungan", "helpdesk", "responsif", "sistem tiket", "umpan balik", "pemecahan masalah", "pelanggan", "servis desk", "obrolan langsung", "respon", "sla", "bantuan", "penyelesaian masalah", "panduan"],
    "Content": ["konten", "video", "vidio", "viral", "tren", "tantangan", "negatif", "dewasa", "edukasi", "musik", "rekomendasi", "humor", "informasi", "berita", "mode", "kecantikan", "permainan", "kuliner", "tutorial", "ulasan", "unboxing", "cerita", "blog", "artikel", "headline", "infografis", "siaran", "reaksi", "parodi", "kolaborasi", "menarik", "baru", "pendek", "panjang", "lucu", "instruktif", "hiburan", "review", "pendidikan", "dokumenter", "podcast", "siaran langsung", "shorts", "sorotan", "sinematik", "klip viral"],
    "Ads/Monetization": ["iklan", "berbayar", "koin", "hadiah", "dana", "penghasilan", "monetisasi", "sponsor", "promo", "langganan", "konten", "voucher", "kupon", "cashback", "harga", "ongkir", "pengiriman", "resi", "pelacakan", "kurir", "paket", "retur", "refund", "penukaran", "bayar", "pembayaran", "transfer", "saldo", "dompet", "invoice", "tagihan", "garansi", "status", "preorder", "cod", "kartu", "debit", "ewallet", "flash", "wishlist", "pengiriman", "pelacakan", "pengiriman paket", "pesanan", "penjual", "toko", "keranjang belanja", "proses checkout", "promosi"],
    "Algorithm": ["rekomendasi", "algoritma", "tidak", "naik", "penonton", "suka", "pengikut", "interaksi", "jangkauan", "tayangan", "personal", "kurasi", "peringkat", "umpan", "jelajahi", "saran", "tren", "bayangan", "analisis", "wawasan", "pertumbuhan", "statistik", "visibilitas", "relevan", "serupa", "populer", "trending", "disesuaikan", "prioritas", "umpan", "rekomendasi pribadi", "pembelajaran mesin", "AI", "peringkat", "personalisasi", "gelembung filter", "bias"],
    "Connectivity": ["internet", "jaringan", "wifi", "sinyal", "putus", "offline", "koneksi", "seluler", "latensi", "kecepatan", "stabilitas", "ping", "hotspot", "bandwidth", "gangguan", "hilang", "lambat", "sambungan", "tidak stabil", "terputus", "mode offline", "online", "drop sinyal", "jaringan", "cakupan", "data", "konektivitas"],
    "Audio": ["suara", "audio", "musik", "volume", "lirik", "lagu", "rekaman", "mikrofon", "gangguan", "penyeimbang", "headphone", "speaker", "bas", "treble", "loop", "hening", "jelas", "sinkronisasi", "efek", "karaoke", "pecah", "hilang", "lambat", "mic", "suara", "derau", "musik latar", "suara", "earphone", "distorsi", "umpan balik", "putar ulang", "equalizer", "level audio"],
    "Notification": ["notifikasi", "pemberitahuan", "tidak", "peringatan", "pengingat", "popup", "pembaruan", "lencana", "suara", "getar", "pesan", "pengaturan", "senyap", "tertunda", "kesalahan", "frekuensi", "pengingat", "peringatan", "push", "lencana", "informasi", "peringatan", "jadwal", "nada dering", "getar", "notifikasi"],
    "Access": ["masuk", "daftar", "verifikasi", "kata", "sandi", "nomor", "otp", "registrasi", "akun", "sosial", "lupa", "atur", "ulang", "autentikasi", "sidik jari", "wajah", "cepat", "sekali", "pulihkan", "mudah", "login", "masuk", "daftar", "atur ulang kata sandi", "kredensial", "token", "buka kunci", "login cepat"],
    "Community": ["komen", "balas", "ikuti", "bagikan", "lapor", "grup", "teman", "lingkaran", "tanda", "reaksi", "sebut", "diskusi", "forum", "obrolan", "kolaborasi", "interaksi", "posting", "permintaan", "interaksi", "aturan", "komunitas", "suka", "bagikan", "mention", "ikuti", "balas", "thread", "sosial", "diskusi", "umpan balik", "interaksi", "moderasi"],
    "Storage": ["memori", "penyimpanan", "file", "boros", "cache", "unduh", "pembaruan", "optimalisasi", "kompresi", "cadangan", "sinkronisasi", "sementara", "pembersihan", "ruang", "aplikasi", "terbatas", "efisien", "penyimpanan", "disk", "kapasitas", "awan", "simpan", "cadangan", "arsip", "bebaskan", "optimalkan", "manajemen data"],
    "TiktokShop": ["belanja", "checkout", "keranjang", "produk", "diskon", "promo", "voucher", "kupon", "cashback", "harga", "ongkir", "pengiriman", "resi", "pelacakan", "kurir", "paket", "retur", "refund", "penukaran", "bayar", "pembayaran", "transfer", "saldo", "dompet", "invoice", "tagihan", "garansi", "status", "preorder", "cod", "kartu", "debit", "ewallet", "flash", "wishlist", "pengiriman", "pelacakan", "pengiriman paket", "pesanan", "penjual", "toko", "keranjang belanja", "proses checkout", "promosi"]
}

TOPIC_DISPLAY_ORDER = [
    "Masalah Teknis & Bug",
    "Masalah Akun & Blokir",
    "Hiburan & Positif",
    "Fitur & Konten",
]

def sort_topics_by_display_order(topics):
    """Keep topic filters in the fixed order used by the dashboard."""
    order_index = {topic: index for index, topic in enumerate(TOPIC_DISPLAY_ORDER)}
    return sorted(topics, key=lambda topic: order_index.get(topic, len(order_index)))

# ==========================================
# 4. DATA LOADING FUNCTIONS
# ==========================================

# --- A. Loader for Topic Analysis ---
@st.cache_data
def load_and_preprocess_data(filepath):
    try:
        if not os.path.exists(filepath):
            return pd.DataFrame()

        df = pd.read_csv(filepath)
        
        # Paksa nama kolom menjadi lowercase biar aman dari typo huruf besar/kecil
        df.columns = [c.lower().strip() for c in df.columns]
        
        if 'at' not in df.columns:
            st.error(f"Column 'at' not found in {filepath}!")
            return pd.DataFrame()
            
        df['at'] = pd.to_datetime(df['at'], errors='coerce')
        df = df.dropna(subset=['at'])
        if df.empty: return pd.DataFrame()
        
        df['year_month'] = df['at'].dt.to_period('M')
        df['month_name'] = df['at'].dt.strftime('%B %Y')
        
        if 'sentiment_nb' in df.columns:
            df['Sentiment'] = df['sentiment_nb']
        else:
            st.error("Column 'Sentiment_NB' missing!")
            return pd.DataFrame()

        # --- DYNAMIC AUTO-DETECTION ENGINE UNTUK MAPPING TOPIK ---
        topic_col = None
        if 'topic' in df.columns:
            topic_col = 'topic'
        elif 'topic_id' in df.columns:
            topic_col = 'topic_id'

        if topic_col:
            # Mengambil digit angka dari isi data kolom topic
            df['raw_num'] = df[topic_col].astype(str).str.extract(r'(\d+)').astype(float)
            unique_nums = df['raw_num'].dropna().unique()
            
            # Deteksi otomatis basis data (0-indexed atau 1-indexed)
            if 0 in unique_nums or (3 in unique_nums and 4 not in unique_nums):
                # Data Berbasis 0 (0, 1, 2, 3)
                df['topic'] = df['raw_num'].map({
                    0: "Masalah Teknis & Bug",
                    1: "Masalah Akun & Blokir",
                    2: "Hiburan & Positif",
                    3: "Fitur & Konten"
                })
            else:
                # Data Berbasis 1 (1, 2, 3, 4)
                df['topic'] = df['raw_num'].map({
                    1: "Masalah Teknis & Bug",
                    2: "Masalah Akun & Blokir",
                    3: "Hiburan & Positif",
                    4: "Fitur & Konten"
                })
            df = df.drop(columns=['raw_num'])
            
        else:
            # Jika menggunakan multi-kolom probabilitas (topic_0, topic_1, dst)
            prob_cols = [c for c in ['topic_0', 'topic_1', 'topic_2', 'topic_3'] if c in df.columns]
            if prob_cols:
                df['raw_topic'] = df[prob_cols].idxmax(axis=1)
                df['topic'] = df['raw_topic'].str.extract(r'(\d+)').astype(float).map({
                    0: "Masalah Teknis & Bug",
                    1: "Masalah Akun & Blokir",
                    2: "Hiburan & Positif",
                    3: "Fitur & Konten"
                })
                df = df.drop(columns=['raw_topic'])

        df['topic'] = df['topic'].fillna("Unknown Topic")
        return df
    except Exception as e:
        st.error(f"Error loading topic data: {e}")
        return pd.DataFrame()

# --- B. Loader & Processing for Aspect Analysis ---
def detect_aspect_sentiment(text, sentiment):
    """Detects aspects in text and pairs them with sentiment."""
    results = []
    if isinstance(text, str):
        text_lower = text.lower()
        for aspect, keywords in ASPEK_DICT.items():
            if any(word in text_lower for word in keywords):
                results.append(f"{aspect} ({sentiment})")
    return list(set(results))

@st.cache_data
def process_aspect_data(filepath):
    """Processes specific data for aspect analysis."""
    if not os.path.exists(filepath):
        return None, None
        
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.lower().strip() for c in df.columns]
        
        if 'at' not in df.columns: return None, None
        if 'content_clean' not in df.columns or 'sentiment' not in df.columns: return None, None
        
        df['at'] = pd.to_datetime(df['at'], errors='coerce')
        
        df["aspek_sentimen"] = df.apply(
            lambda row: detect_aspect_sentiment(row["content_clean"], row["sentiment"]),
            axis=1
        )
        
        df["aspek_sentimen_str"] = df["aspek_sentimen"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else ""
        )
        
        df_exploded = df.explode('aspek_sentimen')
        df_exploded = df_exploded.dropna(subset=['aspek_sentimen'])
        
        df_exploded[['Aspek', 'Sentimen_Raw']] = (
            df_exploded['aspek_sentimen'].str.extract(r'(.*) \((.*)\)')
        )
        
        df_exploded['Sentimen_Label'] = df_exploded['Sentimen_Raw'].astype(float).map({
            0.0: 'Negative',
            1.0: 'Positive'
        })
        
        df_exploded['Bulan_Str'] = df_exploded['at'].dt.to_period('M').astype(str)
        
        return df, df_exploded
        
    except Exception as e:
        st.error(f"Error processing aspect data: {e}")
        return None, None

# ==========================================
# 5. FEATURE PAGES
# ==========================================

def data_page(df):
    """Main Page: Global Sentiment Distribution."""
    st.header("📈 Global Sentiment Distribution")
    st.markdown("Statistical summary of sentiment from all data.")
    
    st.sidebar.divider()
    st.sidebar.subheader("Filter Data")
    unique_months = sorted(df['month_name'].unique().tolist())
    selected_months = st.sidebar.multiselect(
        "Select Month:", options=unique_months, default=unique_months, key="data_filter"
    )
    
    if not selected_months:
        st.warning("Please select a month in the sidebar.")
        return
        
    df_filtered = df[df['month_name'].isin(selected_months)]
    
    total = len(df_filtered)
    counts = df_filtered['Sentiment'].value_counts()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reviews", f"{total:,}")
    if 'Positif' in counts:
        c2.metric("Positive", f"{counts['Positif']:,}", f"{(counts['Positif']/total)*100:.1f}%")
    if 'Negatif' in counts:
        c3.metric("Negative", f"{counts['Negatif']:,}", f"{(counts['Negatif']/total)*100:.1f}%")

    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.subheader("Sentiment Proportion")
        fig = px.pie(names=counts.index, values=counts.values, 
                     color_discrete_map={'Positif':'green', 'Negatif':'red'})
        st.plotly_chart(fig, use_container_width=True)
        
    with c_chart2:
        st.subheader("Sentiment by Topic")
        df_top = df_filtered.groupby('topic')['Sentiment'].value_counts().unstack(fill_value=0).reset_index()
        cols_available = [c for c in ['Positif', 'Negatif'] if c in df_top.columns]
        df_top['Total'] = df_top[cols_available].sum(axis=1)
        df_top = df_top.sort_values('Total', ascending=False)
        
        df_melted = df_top.melt(id_vars='topic', value_vars=cols_available, var_name='Sentiment', value_name='Jml')
        
        fig2 = px.bar(df_melted, x='topic', y='Jml', color='Sentiment', 
                      color_discrete_map={'Positif':'green', 'Negatif':'red'})
        
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

def home_page(df):
    """Topic Detail Page."""
    st.header("🏡 Detailed Topic Analysis")
    
    st.sidebar.divider()
    st.sidebar.subheader("Filter Topics")
    topics = sort_topics_by_display_order(df['topic'].dropna().unique().tolist())
    sel_topic = st.sidebar.multiselect("Select Topic:", options=topics, default=[topics[0]] if topics else [])
    months = sorted(df['month_name'].unique().tolist())
    sel_month = st.sidebar.multiselect("Select Month:", options=months, default=months, key="topic_filter_month")
    
    if not sel_topic or not sel_month:
        st.warning("Please select topics and months.")
        return

    df_f = df[(df['topic'].isin(sel_topic)) & (df['month_name'].isin(sel_month))]
    
    if not df_f.empty:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔢 Data Distribution (Filtered)")
        topic_counts = df_f['topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        st.sidebar.dataframe(topic_counts, hide_index=True, use_container_width=True)

    if df_f.empty: 
        st.info("No data available.")
        return

    fig = px.bar(df_f.groupby(['topic', 'Sentiment']).size().reset_index(name='Jml'), 
                 x='Sentiment', y='Jml', color='Sentiment', facet_col='topic',
                 color_discrete_map={'Positif':'green', 'Negatif':'red'})
    
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1][:30]+"...")) 
    st.plotly_chart(fig, use_container_width=True)

def about_page(df):
    """Time Trend Page."""
    st.header("⏰ Time Trend Analysis")
    
    st.sidebar.divider()
    st.sidebar.subheader("Global Time Filter")
    months = sorted(df['month_name'].unique().tolist())
    sel_month = st.sidebar.multiselect("Select Month:", options=months, default=months, key="time_filter")
    
    if not sel_month: 
        st.warning("Please select a month.")
        return
    
    df_f = df[df['month_name'].isin(sel_month)]

    if not df_f.empty:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔢 Data Count per Topic")
        topic_counts_trend = df_f['topic'].value_counts().reset_index()
        topic_counts_trend.columns = ['Topic', 'Count']
        st.sidebar.dataframe(topic_counts_trend, hide_index=True, use_container_width=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Global Sentiment Trend", "🔥 Topic Popularity Trend", "🎯 Specific Trend per Topic"])

    with tab1:
        st.subheader("Daily Sentiment Movement")
        daily = df_f.groupby([df_f['at'].dt.date, 'Sentiment']).size().reset_index(name='Jml')
        fig1 = px.line(daily, x='at', y='Jml', color='Sentiment', 
                      color_discrete_map={'Positif':'green', 'Negatif':'red'}, markers=True)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.subheader("Topic Popularity Movement")
        daily_topic = df_f.groupby([df_f['at'].dt.date, 'topic']).size().reset_index(name='Jml')
        fig2 = px.line(daily_topic, x='at', y='Jml', color='topic', markers=True)
        fig2.update_layout(hovermode="x unified", legend=dict(yanchor="top", y=-0.2, xanchor="left", x=0))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Deep Dive: Specific Topic & Sentiment")
        c1, c2 = st.columns(2)
        topic_opt = sort_topics_by_display_order(df_f['topic'].dropna().unique().tolist())
        choosen_topic = c1.selectbox("Select Topic:", topic_opt)
        choosen_sentiment = c2.radio("Select Sentiment:", ["Positif", "Negatif"], horizontal=True)
        
        df_detail = df_f[(df_f['topic'] == choosen_topic) & (df_f['Sentiment'] == choosen_sentiment)]
        
        if not df_detail.empty:
            daily_detail = df_detail.groupby(df_detail['at'].dt.date).size().reset_index(name='Jml')
            color_map = 'green' if choosen_sentiment == 'Positif' else 'red'
            fig3 = px.line(daily_detail, x='at', y='Jml', markers=True)
            fig3.update_traces(line_color=color_map)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No data found for this combination.")

def wordcloud_page(df):
    """Word Cloud Visualization Page."""
    st.header("☁️ Word Cloud Analysis")
    
    st.sidebar.divider()
    st.sidebar.subheader("Word Cloud Filter")
    months = sorted(df['month_name'].unique().tolist())
    sel_month = st.sidebar.multiselect("Select Month:", options=months, default=months, key="wc_month")
    sentiment_opt = st.sidebar.radio("Select Sentiment:", ["Positif", "Negatif", "Combined (All)"], key="wc_sentiment")
    
    if not sel_month:
        st.warning("Please select at least one month in the sidebar.")
        return

    df_wc = df[df['month_name'].isin(sel_month)]
    
    if sentiment_opt == "Positif":
        df_wc = df_wc[df_wc['Sentiment'] == 'Positif']
        colormap_style = "Blues"
    elif sentiment_opt == "Negatif":
        df_wc = df_wc[df_wc['Sentiment'] == 'Negatif']
        colormap_style = "Reds"
    else:
        colormap_style = "viridis"

    if df_wc.empty:
        st.info(f"No review data for {sentiment_opt} in the selected month.")
        return

    text_col = 'final_text' if 'final_text' in df.columns else 'content'
    all_text = " ".join(df_wc[text_col].astype(str).tolist())
    
    if not all_text.strip():
        st.warning("Text data is empty, cannot generate Word Cloud.")
        return

    wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap_style, min_font_size=10).generate(all_text)

    st.subheader(f"Word Cloud: {sentiment_opt}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
    
    with st.expander(f"View Review Samples ({sentiment_opt})"):
        st.dataframe(df_wc[[text_col, 'Sentiment']].head(10), use_container_width=True)

def aspect_analysis_page():
    """Aspect Analysis Page."""
    st.header("📱 Aspect-Based Sentiment Analysis (TikTok)")
    st.markdown("Analysis of UI/UX, Features, Performance, etc. based on aspect data.")

    if not os.path.exists(CSV_ASPEK_PATH):
        st.error(f"File '{CSV_ASPEK_PATH}' not yet available. Attempting to download...")
        return

    with st.spinner('Processing aspects and sentiments...'):
        df_proc, df_exp = process_aspect_data(CSV_ASPEK_PATH)
    
    if df_proc is None or df_exp is None:
        st.error("Failed to process aspect data.")
        return

    st.sidebar.divider()
    st.sidebar.subheader("Filter Aspects")
    
    list_bulan = sorted(df_exp['bulan_str'].unique())
    selected_months = st.sidebar.multiselect(
        "Select Aspect Month:", 
        options=list_bulan, 
        default=list_bulan,
        key="aspect_month_selector"
    )
    
    tab1, tab2 = st.tabs(["📊 Overall Analysis", "📈 Monthly Filter Analysis"])
    
    with tab1:
        st.subheader("Sentiment Distribution per Aspect (Total)")
        total_counts = df_exp.groupby(['aspek', 'sentimen_label']).size().reset_index(name='Count')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=total_counts, x="aspek", y="Count", hue="sentimen_label", 
                    palette={'Negative': '#ff6b6b', 'Positive': '#51cf66'}, ax=ax)
        ax.set_title("Total Reviews per Aspect and Sentiment")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

    with tab2:
        if selected_months:
            st.subheader(f"Aspect Analysis for: {', '.join(selected_months)}")
            
            monthly_data = df_exp[df_exp['bulan_str'].isin(selected_months)]
            monthly_counts = monthly_data.groupby(['aspek', 'sentimen_label']).size().reset_index(name='Count')
            
            if not monthly_counts.empty:
                fig_m, ax_m = plt.subplots(figsize=(12, 6))
                sns.barplot(data=monthly_counts, x="aspek", y="Count", hue="sentimen_label", 
                            palette={'Negative': '#ff6b6b', 'Positive': '#51cf66'}, ax=ax_m)
                ax_m.set_title(f"Combined Sentiment Distribution ({', '.join(selected_months)})")
                ax_m.set_xticklabels(ax_m.get_xticklabels(), rotation=45, ha="right")
                ax_m.grid(axis='y', linestyle='--', alpha=0.5)
                st.pyplot(fig_m)
            else:
                st.warning("No aspect data found for the selected months.")
        else:
            st.warning("Please select a month in the sidebar.")

def welcome_page():
    st.title("About the App")
    st.info("TikTok Sentiment Analysis Dashboard (Topics, Trends, and UI/UX Aspects).")

# ==========================================
# 6. MAIN LOGIC (NAVIGATION)
# ==========================================

# Load Topic Data
df_topic = load_and_preprocess_data(CSV_TOPIK_PATH)

# Menu Definitions
MENU_OPTIONS = {
    "📈 Sentiment Distribution": "data",
    "🏡 Topic Analysis": "home",
    "⏰ Time Trend": "about",
    "☁️ Word Cloud": "wordcloud",
    "📱 ABSA (Aspect Analysis)": "aspect",
    "ℹ️ App Info": "welcome"
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Select Page:", list(MENU_OPTIONS.keys()))
page = MENU_OPTIONS[selection]

# Page Routing
if page == "aspect":
    aspect_analysis_page()
elif page == "welcome":
    welcome_page()
else:
    if not df_topic.empty:
        if page == "data":
            data_page(df_topic)
        elif page == "home":
            home_page(df_topic)
        elif page == "about":
            about_page(df_topic)
        elif page == "wordcloud":
            wordcloud_page(df_topic)
    else:
        st.warning(f"Waiting for topic data ('{CSV_TOPIK_PATH}')... or file not yet available.")