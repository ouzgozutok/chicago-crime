#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' 

echo -e "${BLUE}⚡ Kafka Konfigürasyonu Başlatılıyor...${NC}"

# Kafka konteynerının aktif olup olmadığını kontrol et
if [ ! "$(docker ps -q -f name=kafka)" ]; then
    echo -e "${RED}❌ HATA: Kafka konteynerı çalışmıyor! Önce docker-compose up -d çalıştırmalısınız.${NC}"
    exit 1
fi

# crimes topic'ini oluştur (eğer yoksa)
echo -e "${BLUE}📡 'crimes' topic'i oluşturuluyor (Partitions: 3)...${NC}"
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --create --topic crimes --partitions 3 --replication-factor 1 --if-not-exists

# Topic listesini doğrula
echo -e "${GREEN}✅ Aktif Kafka Topicleri:${NC}"
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --list

echo -e "${GREEN}🎉 Kafka kurulumu başarıyla tamamlandı! Akışa hazır.${NC}"