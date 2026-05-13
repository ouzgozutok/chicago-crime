import csv
import json
import time
import os
from kafka import KafkaProducer

KAFKA_BROKER = 'localhost:9092'
TOPIC_NAME = 'crimes'
CSV_FILE_PATH = os.path.join('data', 'chicago_crimes.csv') 

print("🚀 Kafka Producer Başlatılıyor...")

try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        retries=5,
        acks='all'
    )
    print("✅ Kafka Broker bağlantısı başarıyla kuruldu.")
except Exception as e:
    print(f"❌ Kafka Broker'a bağlanılamadı: {e}")
    exit(1)

if not os.path.exists(CSV_FILE_PATH):
    print(f"❌ HATA: '{CSV_FILE_PATH}' konumunda veri seti bulunamadı!")
    print("Lütfen Chicago Crime CSV dosyasını 'data/' klasörünün içine yerleştirin.")
    exit(1)

print(f"📂 Veri seti okunuyor: {CSV_FILE_PATH}")

try:
    with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        counter = 0
        print("\n🔥 Canlı veri akışı başlatıldı! Veriler Kafka'ya gönderiliyor...\n")
        
        for row in csv_reader:
            crime_data = {
                "ID": row.get("ID"),
                "Case_Number": row.get("Case Number"),
                "Date": row.get("Date"),
                "Block": row.get("Block"),
                "IUCR": row.get("IUCR"),
                "Primary_Type": row.get("Primary Type"),
                "Description": row.get("Description"),
                "Location_Description": row.get("Location Description"),
                "Arrest": row.get("Arrest"),
                "Domestic": row.get("Domestic"),
                "Beat": row.get("Beat"),
                "District": row.get("District"),
                "Ward": row.get("Ward"),
                "Community_Area": row.get("Community Area"),
                "FBI_Code": row.get("FBI Code"),
                "X_Coordinate": row.get("X Coordinate"),
                "Y_Coordinate": row.get("Y Coordinate"),
                "Year": row.get("Year"),
                "Updated_On": row.get("Updated On"),
                "Latitude": row.get("Latitude"),
                "Longitude": row.get("Longitude")
            }
            
            producer.send(TOPIC_NAME, value=crime_data)
            counter += 1
            print(f"📡 [GÖNDERİLDİ #{counter}] ID: {crime_data['ID']} | Tip: {crime_data['Primary_Type']} | Bölge: District {crime_data['District']}")
            time.sleep(0.5)
            
except KeyboardInterrupt:
    print("\n🛑 Kullanıcı tarafından durduruldu. Akış kesiliyor...")
except Exception as e:
    print(f"❌ Akış sırasında bir hata oluştu: {e}")
finally:
    producer.flush()
    producer.close()
    print("🔌 Kafka bağlantısı kapatıldı. Producer sonlandırıldı.")