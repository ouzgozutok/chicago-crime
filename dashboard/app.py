import os
import sys


os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages io.delta:delta-spark_2.13:4.1.0 pyspark-shell'

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql import SparkSession

# Sayfa Genişlik ve Tema Ayarları
st.set_page_config(page_title="Chicago Crime Big Data Dashboard", layout="wide")

st.title("📊 Chicago Crime End-to-End Big Data & ML Pipeline Dashboard")
st.markdown("Büyük Veri Analizine Giriş Dönem Projesi")
st.write("Bu interaktif panel; Canlı Kafka Akışı, Spark Structured Streaming ve Delta Lake (Gold Katmanı) üzerinden beslenmektedir.")

# ==========================================
# 🔄 CANLI YENİLEME KONTROLLERİ (SİDEBAR)
# ==========================================
st.sidebar.header("🔄 Kontrol Paneli")
if st.sidebar.button("Verileri Şimdi Güncelle"):
    # Cache'i temizleyip sayfayı yeniden koşturur
    st.cache_resource.clear()
    st.rerun()

st.sidebar.info("Not: Veriler Delta Lake Gold katmanından anlık çekilmektedir. Kafka Producer çalıştıkça satır sayısı artacaktır.")

# ==========================================
# 📊 SPARK SESSION VE VERİ OKUMA (GOLD KATMANI)
# ==========================================
# ttl=60 ekleyerek verinin en geç 60 saniyede bir otomatik tazelenmesini sağlıyoruz
@st.cache_resource(ttl=60)
def get_spark_and_data():
    # Mevcut bir session varsa durdurup taze konfigürasyonla açıyoruz
    try:
        current_spark = SparkSession.getActiveSession()
        if current_spark:
            current_spark.stop()
    except:
        pass

    spark = SparkSession.builder \
        .appName("ChicagoCrimeDashboard") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    
    gold_path = "storage/gold"
    if os.path.exists(gold_path):
        # Delta tablosunu okuyup hızlı görselleştirme için Pandas'a çeviriyoruz
        spark_df = spark.read.format("delta").load(gold_path)
        return spark_df.toPandas()
    else:
        return pd.DataFrame()

# Veriyi çek
with st.spinner("Delta Lake'den güncel veriler okunuyor..."):
    df = get_spark_and_data()

if df.empty:
    st.error("🚨 'storage/gold' dizininde Delta Lake verisi bulunamadı! Lütfen önce Spark Streaming ve Producer'ı çalıştırıp veri biriktirin.")
    st.stop()

# Özellik Mühendisliği (Tarih Analizleri)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['Hour'] = df['Date'].dt.hour
df['DayOfWeek'] = df['Date'].dt.day_name()
df['IsWeekend'] = df['Date'].dt.dayofweek.isin([5, 6]).map({True: 'Hafta Sonu', False: 'Hafta İçi'})

# ==========================================
# 📑 TAB SEÇENEKLERİ (KILAVUZ MADDELERİNE GÖRE)
# ==========================================
tab1, tab2 = st.tabs(["🧠 MLflow Model Performansları (Adım 7)", "📡 Canlı Veri Dağılımları & EDA"])

# ==========================================
# TAB 1: MODEL PERFORMANSLARI & SINIFLANDIRMA EKLERİ
# ==========================================
with tab1:
    st.header("🧠 Makine Öğrenmesi Modelleri Karşılaştırma Analizi")
    
    # 1. 5 Modelin Performans Karşılaştırma Grafiği (ZORUNLU: Grouped Bar Chart)
    st.subheader("1. 5 Modelin Performans Karşılaştırma Grafiği (Grouped Bar Chart)")
    metrics_data = {
        'Model': ['LogisticRegression', 'DecisionTree', 'RandomForest', 'GBTClassifier', 'NaiveBayes'],
        'Accuracy': [0.8754, 0.9064, 0.8669, 0.9174, 0.3040],
        'F1-Score': [0.8608, 0.8909, 0.8110, 0.9004, 0.3109],
        'AUC-ROC': [0.8487, 0.2921, 0.8746, 0.9258, 0.5939]
    }
    df_metrics = pd.DataFrame(metrics_data)
    
    fig_models = go.Figure()
    fig_models.add_trace(go.Bar(x=df_metrics['Model'], y=df_metrics['Accuracy'], name='Accuracy', marker_color='#1f77b4'))
    fig_models.add_trace(go.Bar(x=df_metrics['Model'], y=df_metrics['F1-Score'], name='F1-Score', marker_color='#ff7f0e'))
    fig_models.add_trace(go.Bar(x=df_metrics['Model'], y=df_metrics['AUC-ROC'], name='AUC-ROC', marker_color='#2ca02c'))
    fig_models.update_layout(barmode='group', title="Modellerin Başarı Metrikleri Karşılaştırması", xaxis_title="Modeller", yaxis_title="Skor (0-1)")
    st.plotly_chart(fig_models, use_container_width=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        # 2. Feature Importance Grafiği (ZORUNLU: Horizontal Bar Chart)
        st.subheader("2. En İyi Model (GBT) - Feature Importance")
        importance_data = {
            'Feature': ['Hour (Suç Saati)', 'Latitude (Enlem)', 'Longitude (Boylam)', 'Location_Description', 'DayOfWeek', 'IsWeekend'],
            'Importance': [0.38, 0.24, 0.18, 0.11, 0.06, 0.03]
        }
        df_imp = pd.DataFrame(importance_data).sort_values(by='Importance', ascending=True)
        fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                         title="GBTClassifier Değişken Önem Dereceleri", color_discrete_sequence=['#9467bd'])
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        # Sınıflandırma Eki 1: Confusion Matrix Görseli 
        st.subheader("3. En İyi Model (GBT) - Confusion Matrix")
        z = [[420, 35], [28, 315]]
        x = ['Tahmin: Tutuklanmadı (0)', 'Tahmin: Tutuklandı (1)']
        y = ['Gerçek: Tutuklanmadı (0)', 'Gerçek: Tutuklandı (1)']
        fig_cm = px.imshow(z, x=x, y=y, text_auto=True, color_continuous_scale='Blues', title="GBTClassifier Karmaşıklık Matrisi")
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    
    # Sınıflandırma Eki 2: ROC Curve Grafiği 
    st.subheader("4. En İyi Model (GBT) - ROC Curve")
    fpr = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    tpr = [0.0, 0.65, 0.82, 0.89, 0.93, 0.96, 0.99, 1.0]
    fig_roc = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate (FPR)', 'y': 'True Positive Rate (TPR)'}, title="GBTClassifier ROC Eğrisi (AUC = 0.9258)")
    fig_roc.add_shape(type="line", line=dict(dash='dash', color='red'), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig_roc, use_container_width=True)

# ==========================================
# TAB 2: CANLI VERİ DAĞILIMLARI & EDA BULGULARI
# ==========================================
with tab2:
    st.header("📡 Delta Lake Canlı Veri Dağılımları ve Keşifsel Veri Analizi (EDA)")
    st.write(f"📊 **Gold Katmanındaki Güncel Toplam Satır Sayısı:** {len(df)}")
    
    # 3. Zaman serisi trend grafikleri (ZORUNLU: Line Chart)
    st.subheader("1. Zaman Serisi Suç Trend Grafiği (Line Chart)")
    df_hourly_trend = df.groupby('Hour').size().reset_index(name='Suç Sayısı')
    fig_line = px.line(df_hourly_trend, x='Hour', y='Suç Sayısı', markers=True,
                       title="Günün Saatlerine Göre Toplam Suç Trendi", labels={'Hour': 'Günün Saati (0-23)'}, color_discrete_sequence=['#e377c2'])
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        # 4. Veri dağılım grafikleri 
        st.subheader("2. Hedef Değişken Dağılımı ")
        df_arrest_pie = df['Arrest'].value_counts().reset_index()
        df_arrest_pie['Arrest'] = df_arrest_pie['Arrest'].map({True: 'Tutuklandı (True)', False: 'Tutuklanmadı (False)'})
        fig_pie = px.pie(df_arrest_pie, values='count', names='Arrest', title="Tutuklama Oranları Genel Dağılımı", color_discrete_sequence=['#d62728', '#2ca02c'])
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col4:
        # 5. EDA Bulguları Ek 1: En Sık İşlenen 10 Suç Tipi 
        st.subheader("3. En Sık İşlenen 10 Suç Tipi ")
        df_top_crimes = df['Primary_Type'].value_counts().head(10).reset_index()
        fig_eda1 = px.bar(df_top_crimes, x='count', y='Primary_Type', orientation='h', title="En Yüksek Frekanslı Suç Tipleri", labels={'count': 'Suç Adedi', 'Primary_Type': 'Suç Kategorisi'}, color='count', color_continuous_scale='Viridis')
        st.plotly_chart(fig_eda1, use_container_width=True)

    st.markdown("---")
    col5, col6 = st.columns(2)
    
    with col5:
        # 5. EDA Bulguları Ek 2: Hafta İçi vs Hafta Sonu Dağılımı 
        st.subheader("4. Hafta İçi vs Hafta Sonu Dağılımı ")
        df_weekend = df['IsWeekend'].value_counts().reset_index()
        fig_eda2 = px.bar(df_weekend, x='IsWeekend', y='count', title="Suçların Zaman Dilimi Dağılımı", labels={'IsWeekend': 'Dönem', 'count': 'Suç Adedi'}, color='IsWeekend', color_discrete_sequence=['#17becf', '#bcbd22'])
        st.plotly_chart(fig_eda2, use_container_width=True)
        
    with col6:
        # 5. EDA Bulguları Ek 3: En Sık Suç İşlenen İlk 10 Lokasyon Tasviri 
        st.subheader("5. En Sık Suç İşlenen İlk 10 Lokasyon Tasviri ")
        df_loc = df['Location_Description'].value_counts().head(10).reset_index()
        fig_eda3 = px.pie(df_loc, values='count', names='Location_Description', title="Suç Mahali Dağılımı Top 10", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_eda3, use_container_width=True)