# 🚀 Real-time Chat Kurulum Rehberi

## 📋 Gereksinimler

### 1. **Redis Kurulumu**
```bash
# Windows için (WSL veya Docker önerilir)
# WSL Ubuntu'da:
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server

# macOS için:
brew install redis
brew services start redis

# Linux için:
sudo apt-get install redis-server
sudo systemctl start redis-server
```

### 2. **Python Paketleri**
```bash
pip install -r requirements.txt
```

## 🔧 Kurulum Adımları

### 1. **Veritabanı Migration'ları**
```bash
python manage.py makemigrations
python manage.py migrate
```

### 2. **Redis Bağlantısını Test Et**
```bash
# Redis CLI ile test
redis-cli ping
# "PONG" yanıtı almalısınız
```

### 3. **Sunucuyu Başlat**
```bash
# Daphne ile ASGI sunucusu başlat
daphne cancerDiagnosisProject.asgi:application

# Veya development için:
python manage.py runserver
```

## 🎯 Özellikler

### ✅ **Eklenen Özellikler:**

1. **Real-time Mesajlaşma**
   - WebSocket bağlantısı
   - Anlık mesaj gönderme/alma
   - Bağlantı durumu göstergesi

2. **AI Bot Yanıtları**
   - Keyword-based yanıtlar
   - Deri hastalıkları hakkında bilgi
   - Teşhis yorumlama desteği

3. **Kullanıcı Deneyimi**
   - Modern chat arayüzü
   - Hızlı yanıt butonları
   - Mesaj geçmişi
   - Responsive tasarım

4. **Admin Paneli**
   - Chat odalarını yönetme
   - Mesajları görüntüleme
   - Kullanıcı oturumlarını takip

## 🔗 URL'ler

- **Ana Sayfa:** `/`
- **Chat Sayfası:** `/chat/`
- **Hastane Arama:** `/hospitals/`
- **Admin Panel:** `/admin/`

## 🛠️ Geliştirme

### **Chat Bot Yanıtlarını Özelleştirme:**
`cancerDiagnosisApp/consumers.py` dosyasındaki `generate_bot_response` fonksiyonunu düzenleyin.

### **Yeni Özellikler Eklemek:**
1. **Dosya Paylaşımı:** WebSocket üzerinden resim gönderme
2. **Sesli Mesajlar:** Audio recording ve playback
3. **Video Görüşme:** WebRTC entegrasyonu
4. **Çoklu Dil:** Internationalization desteği

## 🚨 Sorun Giderme

### **Redis Bağlantı Hatası:**
```python
# settings.py'de Redis ayarlarını kontrol edin
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
        },
    },
}
```

### **WebSocket Bağlantı Hatası:**
- Tarayıcı konsolunu kontrol edin
- Firewall ayarlarını kontrol edin
- Daphne sunucusunun çalıştığından emin olun

### **Mesajlar Görünmüyor:**
- Veritabanı migration'larını kontrol edin
- Admin panelinde mesajları kontrol edin
- Redis bağlantısını test edin

## 📊 Performans

### **Önerilen Ayarlar:**
- **Redis:** 512MB RAM
- **Django:** 4 worker process
- **WebSocket:** 1000 concurrent connection

### **Monitoring:**
```bash
# Redis memory kullanımı
redis-cli info memory

# Active connections
redis-cli client list
```

## 🔒 Güvenlik

### **Öneriler:**
1. **Rate Limiting:** Mesaj gönderme hızını sınırla
2. **Input Validation:** Mesaj içeriğini doğrula
3. **Authentication:** Kullanıcı girişi ekle
4. **HTTPS:** SSL sertifikası kullan

## 🎉 Başarılı Kurulum!

Chat sistemi artık hazır! Kullanıcılar `/chat/` sayfasından AI asistanla konuşabilir. 