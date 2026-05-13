import os
import sys
import pyspark
import mlflow
import mlflow.spark
import shutil
# Grafik çizimi ve veri manipülasyonu kütüphaneleri (MLflow'a yüklemek için)
import matplotlib
matplotlib.use('Agg')  # Headless (arayüzsüz) terminal desteği için
import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, dayofweek, when, trim
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression, 
    DecisionTreeClassifier, 
    RandomForestClassifier, 
    GBTClassifier, 
    NaiveBayes
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# =====================================================================
# 🛠️ DİNAMİK VERSİYON VE SCALA TESPİT MOTORU
# =====================================================================
def get_compatible_packages():
    spark_version = pyspark.__version__
    scala_version = "2.13"
    pyspark_dir = os.path.dirname(pyspark.__file__)
    jars_dir = os.path.join(pyspark_dir, "jars")
    
    if os.path.exists(jars_dir):
        for f in os.listdir(jars_dir):
            if f.startswith("spark-core_"):
                parts = f.split("_")
                if len(parts) > 1:
                    scala_part = parts[1].split("-")[0]
                    if scala_part in ["2.12", "2.13"]:
                        scala_version = scala_part
                    ver_part = parts[1].split("-")[1].replace(".jar", "")
                    if ver_part:
                        spark_version = ver_part
    
    delta_artifact = "delta-spark_4.1" if spark_version.startswith("4.1") else "delta-core"
    delta_version = "4.1.0" if spark_version.startswith("4.1") else "2.4.0"
    delta_package = f"io.delta:{delta_artifact}_{scala_version}:{delta_version}"
    
    return delta_package

DELTA_PKG = get_compatible_packages()

# ==========================================
# 1. ML Spark Session Kurulumu
# ==========================================
spark = SparkSession.builder \
    .appName("ChicagoCrimesMLPipeline") \
    .config("spark.jars.packages", DELTA_PKG) \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("✅ [T3] ML Spark Session Delta Lake desteğiyle kuruldu.")

# ==========================================
# 2. Gold Delta Tablosunun Okunması
# ==========================================
GOLD_PATH = "storage/gold"
if not os.path.exists(GOLD_PATH) or len(os.listdir(GOLD_PATH)) == 0:
    print("❌ HATA: 'storage/gold' klasörü boş! Önce 'streaming_etl.py' çalıştırılıp Delta verisi üretilmelidir.")
    sys.exit(1)

print(f"📂 [T3] Gold Delta Lake tablosu okunuyor: {GOLD_PATH}")
raw_gold_df = spark.read.format("delta").load(GOLD_PATH)

# ==========================================
# 3. Özellik Mühendisliği (Feature Engineering - Adım 5)
# ==========================================
# Hoca en az 5 özellik türetilmesini istiyor:
# 1. Hour: Suçun işlendiği saat dilimi (0-23)
# 2. DayOfWeek: Haftanın günü (1=Pazar, 7=Cumartesi)
# 3. IsWeekend: Hafta sonu mu (1) yoksa hafta içi mi (0)
# 4. Latitude (Double Cast): Sayısal coğrafi veri
# 5. Longitude (Double Cast): Sayısal coğrafi veri
# 6. Label: Hedef değişkenimiz olan 'Arrest' boolean alanının 0/1 integer cast hali
print("🧠 [T3] Özellik mühendisliği (Feature Engineering) başlatılıyor...")

ml_df = raw_gold_df \
    .filter(
        (col("Latitude").isNotNull()) & (trim(col("Latitude")) != "") & 
        (col("Longitude").isNotNull()) & (trim(col("Longitude")) != "") &
        (col("Date").isNotNull()) & (trim(col("Date")) != "")
    ) \
    .withColumn("ts", to_timestamp(col("Date"), "MM/dd/yyyy hh:mm:ss a")) \
    .withColumn("Hour", hour(col("ts"))) \
    .withColumn("DayOfWeek", dayofweek(col("ts"))) \
    .withColumn("IsWeekend", when(col("DayOfWeek").isin([1, 7]), 1).otherwise(0)) \
    .withColumn("Latitude", col("Latitude").cast("double")) \
    .withColumn("Longitude", col("Longitude").cast("double")) \
    .withColumn("label", col("Arrest").cast("integer")) \
    .withColumn("Primary_Type", when((col("Primary_Type").isNull()) | (trim(col("Primary_Type")) == ""), "UNKNOWN").otherwise(col("Primary_Type"))) \
    .withColumn("Location_Description", when((col("Location_Description").isNull()) | (trim(col("Location_Description")) == ""), "UNKNOWN").otherwise(col("Location_Description"))) \
    .filter(col("Hour").isNotNull() & col("label").isNotNull())
# ==========================================
# 4. Spark ML Pipeline Hazırlığı
# ==========================================
# Kategorik verileri StringIndexer ve OneHotEncoder ile sayısallaştırıyoruz
indexer_type = StringIndexer(inputCol="Primary_Type", outputCol="Type_Indexed", handleInvalid="keep")
indexer_loc = StringIndexer(inputCol="Location_Description", outputCol="Loc_Indexed", handleInvalid="keep")

ohe = OneHotEncoder(
    inputCols=["Type_Indexed", "Loc_Indexed"],
    outputCols=["Type_Vec", "Loc_Vec"]
)

# Tüm özellikleri tek bir vektörde birleştiriyoruz
feature_cols = ["Hour", "DayOfWeek", "IsWeekend", "Latitude", "Longitude", "Type_Vec", "Loc_Vec"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Veriyi Eğitim (%80) ve Test (%20) olarak ayırıyoruz
train_data, test_data = ml_df.randomSplit([0.8, 0.2], seed=42)

# ==========================================
# 5. MLflow Kurulumu ve Modellerin Eğitimi (Adım 6)
# ==========================================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Chicago_Crime_Arrest_Prediction")

# Eğiteceğimiz 5 Farklı Model Sınıflandırıcısı (Adım 6)
classifiers = {
    "LogisticRegression": LogisticRegression(featuresCol="features", labelCol="label"),
    "DecisionTree": DecisionTreeClassifier(featuresCol="features", labelCol="label"),
    "RandomForest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20),
    "GBTClassifier": GBTClassifier(featuresCol="features", labelCol="label", maxIter=10),
    "NaiveBayes": NaiveBayes(featuresCol="features", labelCol="label", modelType="gaussian")
}

# Değerlendirme Metrik Evaluator'ları
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

print("\n🚀 [T3] MLflow Deney Takibi ve Model Eğitimleri Başlatılıyor...\n")

for model_name, clf in classifiers.items():
    with mlflow.start_run(run_name=model_name):
        print(f"🔄 Eğitim Başladı: {model_name}")
        
        # ML Pipeline kuruyoruz
        pipeline = Pipeline(stages=[indexer_type, indexer_loc, ohe, assembler, clf])
        
        # Modeli eğitiyoruz
        model = pipeline.fit(train_data)
        
        # Test verisi üzerinde tahmin üretiyoruz
        predictions = model.transform(test_data)
        
        # Metrikleri hesaplıyoruz
        accuracy = evaluator_acc.evaluate(predictions)
        f1_score = evaluator_f1.evaluate(predictions)
        
        # NaiveBayes rawPrediction formatı farklı olduğu için AUC hesaplamasını koşullu yapıyoruz
        try:
            auc_score = evaluator_auc.evaluate(predictions)
        except Exception:
            auc_score = 0.5  # Fallback

        # MLflow'a Parametre ve Metrik loglama
        mlflow.log_param("classifier", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1_score)
        mlflow.log_metric("auc_roc", auc_score)
        
        # 📈 Akademik Katma Değer: Feature Importance Çizimi (Sadece RF ve GBT için)
        if model_name in ["RandomForest", "GBTClassifier"]:
            try:
                # Eğitilmiş pipeline'dan sınıflandırıcıyı ve özellikleri alıyoruz
                fit_clf = model.stages[-1]
                importances = fit_clf.featureImportances.toArray()
                
                # İlk 5 temel numerik özelliğin önemini görselleştiriyoruz
                labels = ["Hour", "DayOfWeek", "IsWeekend", "Latitude", "Longitude"]
                y_pos = np.arange(len(labels))
                feat_imp = importances[:5]
                
                plt.figure(figsize=(8, 4))
                plt.barh(y_pos, feat_imp, align='center', color='skyblue', edgecolor='black')
                plt.yticks(y_pos, labels)
                plt.xlabel('Önem Derecesi (Importance Score)')
                plt.title(f'{model_name} - En Önemli 5 Sayısal Özellik')
                plt.tight_layout()
                
                # Grafiği yerel diske kaydet
                plot_path = f"feature_importance_{model_name}.png"
                plt.savefig(plot_path)
                plt.close()
                
                # Grafiği MLflow Artifacts (Bulut deposuna) fırlat!
                mlflow.log_artifact(plot_path)
                os.remove(plot_path)  # Yerel kalıntıyı sil
                print(f"📊 {model_name} için Feature Importance grafiği üretildi ve MLflow'a yüklendi.")
            except Exception as e:
                print(f"⚠️ Feature Importance çıkarılırken hata oluştu: {str(e)}")

        # Eğitilen modeli MLflow Artifact deposuna Spark formatında kaydet
        temp_local_path = f"/tmp/model_{model_name}"
        
        # Eğer önceden kalma geçici klasör varsa temizliyoruz
        if os.path.exists(temp_local_path):
            shutil.rmtree(temp_local_path)
            
        # 1. Modeli tüm metadata'sıyla birlikte yerel diske kaydediyoruz
        mlflow.spark.save_model(spark_model=model, path=temp_local_path)
        
        # 2. Python HTTP istemcisiyle bu yerel klasörü MLflow Docker'ına yüklüyoruz
        mlflow.log_artifacts(temp_local_path, artifact_path=f"model_{model_name}")
        
        # 3. Geçici yerel klasörü temizliyoruz
        shutil.rmtree(temp_local_path)
        # =============================================================
        
        print(f"📈 {model_name} -> Accuracy: {accuracy:.4f} | F1-Score: {f1_score:.4f} | AUC: {auc_score:.4f}")
        print(f"✅ {model_name} başarıyla MLflow paneline kaydedildi.\n")

print("🎉 [T3] Tüm model eğitimleri ve MLflow kayıtları tamamlandı!")