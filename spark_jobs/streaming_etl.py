import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp, when
from pyspark.sql.types import StructType, StructField, StringType

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

    if spark_version.startswith("3.4"):
        delta_artifact = "delta-core"
        delta_version = "2.4.0"
    elif spark_version.startswith("3.5"):
        delta_artifact = "delta-spark"
        delta_version = "3.1.0"
    elif spark_version.startswith("4.0"):
        delta_artifact = "delta-spark"
        delta_version = "4.0.0"
    elif spark_version.startswith("4.1"):
        delta_artifact = "delta-spark_4.1"
        delta_version = "4.1.0"
    else:
        delta_artifact = "delta-spark_4.1"
        delta_version = "4.1.0"
        
    kafka_package = f"org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}"
    delta_package = f"io.delta:{delta_artifact}_{scala_version}:{delta_version}"
    
    return kafka_package, delta_package

KAFKA_PKG, DELTA_PKG = get_compatible_packages()

# ==========================================
# 1. Spark Session Kurulumu
# ==========================================
spark = SparkSession.builder \
    .appName("ChicagoCrimesMedallionETL") \
    .config("spark.jars.packages", f"{KAFKA_PKG},{DELTA_PKG}") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("✅ [T2.2] Spark Session Delta Lake entegrasyonuyla başarıyla kuruldu.")

# ==========================================
# 2. Şema Tanımlama
# ==========================================
crime_schema = StructType([
    StructField("ID", StringType(), True),
    StructField("Case_Number", StringType(), True),
    StructField("Date", StringType(), True),
    StructField("Block", StringType(), True),
    StructField("IUCR", StringType(), True),
    StructField("Primary_Type", StringType(), True),
    StructField("Description", StringType(), True),
    StructField("Location_Description", StringType(), True),
    StructField("Arrest", StringType(), True),
    StructField("Domestic", StringType(), True),
    StructField("Beat", StringType(), True),
    StructField("District", StringType(), True),
    StructField("Ward", StringType(), True),
    StructField("Community_Area", StringType(), True),
    StructField("FBI_Code", StringType(), True),
    StructField("X_Coordinate", StringType(), True),
    StructField("Y_Coordinate", StringType(), True),
    StructField("Year", StringType(), True),
    StructField("Updated_On", StringType(), True),
    StructField("Latitude", StringType(), True),
    StructField("Longitude", StringType(), True)
])

# Klasör Yolları
BASE_STORAGE = "storage"
CHECKPOINTS_DIR = os.path.join(BASE_STORAGE, "checkpoints")

# ==========================================
# 3. Kafka'dan Tek Seferde Akış Okuma (In-Memory Source)
# ==========================================
print("📡 Kafka 'crimes' topic'ine readStream ile bağlanılıyor...")
kafka_stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "crimes") \
    .option("startingOffsets", "latest") \
    .load()

# ==========================================
# 🛡️ BRANCH 1: Bronze Layer (Raw Veri)
# ==========================================
print("📥 Bronze Akışı Hazırlanıyor...")
bronze_df = kafka_stream_df.withColumn("ingested_at", current_timestamp())

# ==========================================
# 🥈 BRANCH 2: Silver Layer (Temizlenmiş & Parsed Veri)
# ==========================================
print("🧹 Silver Akışı Hazırlanıyor...")
string_df = kafka_stream_df.selectExpr("CAST(value AS STRING) as json_payload")
parsed_df = string_df.select(from_json(col("json_payload"), crime_schema).alias("data")).select("data.*")

cleaned_df = parsed_df \
    .filter(col("ID").isNotNull() & col("Date").isNotNull() & col("Primary_Type").isNotNull()) \
    .withColumn("Arrest", when(col("Arrest") == "true", True).otherwise(False)) \
    .withColumn("Domestic", when(col("Domestic") == "true", True).otherwise(False)) \
    .dropDuplicates(["ID"])

# ==========================================
# 🥇 BRANCH 3: Gold Layer (ML Özellikleri)
# ==========================================
print("👑 Gold Akışı Hazırlanıyor...")
gold_df = cleaned_df.select(
    "ID", "Date", "Primary_Type", "Location_Description", 
    "Arrest", "Domestic", "District", "Year", "Latitude", "Longitude"
)

# ==========================================
# 🚀 AKIŞLARI EŞZAMANLI BAŞLATMA (Multi-Sink Start)
# ==========================================
print("\n🔥 Tüm Medallion Katmanları (Bronze, Silver, Gold) Eşzamanlı Ateşleniyor...\n")

# Bronze Sink
bronze_query = bronze_df.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", os.path.join(CHECKPOINTS_DIR, "bronze")) \
    .start(os.path.join(BASE_STORAGE, "bronze"))

# Silver Sink
silver_query = cleaned_df.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", os.path.join(CHECKPOINTS_DIR, "silver")) \
    .start(os.path.join(BASE_STORAGE, "silver"))

# Gold Sink
gold_query = gold_df.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", os.path.join(CHECKPOINTS_DIR, "gold")) \
    .start(os.path.join(BASE_STORAGE, "gold"))

# Konsolda canlı durum takibi yapmak için Console Sink (Opsiyonel)
console_query = gold_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", "false") \
    .start()

# Tüm akışların hayatta kalmasını sağla
spark.streams.awaitAnyTermination()