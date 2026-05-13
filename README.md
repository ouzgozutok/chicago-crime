# 📊 Chicago Crime End-to-End Big Data & ML Pipeline

Bu proje, Chicago şehri suç veri setlerini kullanarak gerçek zamanlı veri akışı, çok katmanlı modern veri ambarı yönetimi (Lakehouse) ve makine öğrenmesi tahminleme süreçlerini içeren uçtan uca bir **Kappa Mimarisi (Kappa Architecture)** uygulamasıdır.

Proje, Kocaeli Üniversitesi Büyük Veri Analizine Giriş dersi kapsamında Dr. Ayşe Gül Eker'in belirlediği 7 adımdan oluşan tüm proje gereksinimlerini eksiksiz karşılamaktadır.

---

## 🏗️ Sistem Mimarisi (Architecture)

Projenin veri hattı (data pipeline) katmanları ve kullanılan teknolojiler aşağıda özetlenmiştir:

```text
[Canlı CSV Verisi] 
       │
       ▼
[Kafka Producer] (ingestion/producer.py)
       │ (Topic: crimes, Partitions: 3)
       ▼
[Spark Structured Streaming] (spark_jobs/streaming_etl.py)
       │
       ├─► 🥉 Bronze Katmanı (Ham JSON Veri diske yazım)
       ├─► 🥈 Silver Katmanı (Parse, Tip dönüşümü, Deduplication)
       └─► 🥇 Gold Katmanı   (Analize hazır veri katmanı)
       │
       ▼
[Spark ML Pipeline] (ml_models/train_model.py)
       │ (Feature Engineering & MLflow Tracking)
       ▼
[Streamlit Dashboard] (dashboard/app.py)
       └─► İnteraktif Görselleştirme (Adım 7)
```

---

## 🛠️ Teknik Detaylar ve Uygulanan Adımlar

### 1. Veri Mühendisliği (Medallion Architecture)
Delta Lake kullanılarak verinin ACID tutarlılığı sağlanmış ve veri 3 aşamada işlenmiştir:
- **Bronze:** Kafka'dan gelen ham mesajlar.
- **Silver:** Şema eşleme, veri temizliği ve `ID` bazlı mükerrer veri elenmesi.
- **Gold:** Makine öğrenmesine hazır, özellik mühendisliği tamamlanmış en temiz katman.

### 2. Özellik Mühendisliği (Feature Engineering)
Suçun işlendiği tarih verisinden aşağıdaki 5+ yeni özellik türetilmiştir:
- `Hour`, `DayOfWeek`, `IsWeekend`, `District_Category`, `Location_Category`.

### 3. Makine Öğrenmesi ve Deney Takibi
Spark ML kütüphanesi ile 5 farklı model eğitilmiş ve **MLflow** ile tüm süreç kayıt altına alınmıştır:
- Logistic Regression, Decision Tree, Random Forest, GBTClassifier, Naive Bayes.

---

## 📈 Adım 7: Görselleştirme ve Dashboard

Proje sonuçları, **Streamlit** ve **Plotly** kullanılarak interaktif bir dashboard'a dönüştürülmüştür.

**Karşılanan Zorunlu Görseller:**
- ✅ **Performans Karşılaştırması:** 5 modelin Accuracy, F1 ve AUC değerlerini içeren Grouped Bar Chart.
- ✅ **Feature Importance:** En iyi model (GBT) için Horizontal Bar Chart.
- ✅ **Zaman Serisi:** Saatlik suç yoğunluğu trendi (Line Chart).
- ✅ **Veri Dağılımı:** Tutuklama oranları (Pie Chart).
- ✅ **Ek EDA Bulguları:** En sık suç işlenen 10 lokasyon, suç tipi frekansları ve hafta içi/sonu analizi.
- ✅ **Sınıflandırma Ekleri:** Confusion Matrix ve ROC Curve görselleri.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Altyapıyı Başlatma (Docker)
```bash
cd docker
docker-compose up -d
```

### 2. Veri Akışı ve ETL
```bash
# Streaming motorunu başlatın
python3 spark_jobs/streaming_etl.py

# Veri akışını (Producer) başlatın
python3 ingestion/producer.py
```

### 3. Model Eğitimi ve Dashboard
```bash
# ML modellerini eğitin
python3 ml_models/train_model.py

# Dashboard'u çalıştırın
streamlit run dashboard/app.py
```

---

## 📊 Özet Sonuçlar

En başarılı modelimiz **GBTClassifier** olmuştur:
- **Doğruluk (Accuracy):** %91.74
- **AUC-ROC:** 0.9258
- **F1-Score:** 0.9004

*Detaylı analizlere ve geçmiş deneylere `http://localhost:5000` (MLflow) ve `http://localhost:8501` (Dashboard) adreslerinden erişilebilir.*
```