from django.shortcuts import render
import os
from django.conf import settings
from cancerDiagnosisApp.train_model import predict_image
from django.http import JsonResponse
import json
import math
import requests
from .models import ChatRoom, ChatMessage, UserSession

def HomePage(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("image")
        if not uploaded_file:
            return render(request, 'HomePage.html', {'error': "Lütfen bir görsel yükleyin."})

        static_dir = os.path.join(settings.STATICFILES_DIRS[0], "img")
        os.makedirs(static_dir, exist_ok=True)
        saved_file_path = os.path.join(static_dir, uploaded_file.name)

        with open(saved_file_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        try:
            predicted_diagnosis = predict_image(saved_file_path)
            return render(request, 'HomePage.html', {
                'predicted_diagnosis': predicted_diagnosis,
                'image_name': uploaded_file.name
            })
        except Exception as e:
            if os.path.exists(saved_file_path):
                os.remove(saved_file_path)
            return render(request, 'HomePage.html', {
                'error': f"Tahmin sırasında hata oluştu: {str(e)}"
            })

    return render(request, 'HomePage.html')

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    İki koordinat arasındaki mesafeyi (km) hesaplar
    """
    R = 6371  # Dünya'nın yarıçapı (km)
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def get_nearby_hospitals(request):
    """
    OpenStreetMap Nominatim API ile en yakın hastaneleri çeker ve cildiye bölümü anahtar kelime araması yapar
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_lat = float(data.get('latitude'))
            user_lng = float(data.get('longitude'))
            max_distance = float(data.get('max_distance', 50))  # km
            limit = int(data.get('limit', 10))

            # Nominatim API ile hastane arama
            url = (
                f'https://nominatim.openstreetmap.org/search?'
                f'q=hospital&format=json&limit={limit*3}&'
                f'viewbox={user_lng-max_distance/111:.6f},{user_lat+max_distance/111:.6f},'
                f'{user_lng+max_distance/111:.6f},{user_lat-max_distance/111:.6f}'
            )
            headers = {'User-Agent': 'DeriHastaligiTeshisi/1.0 (your@email.com)'}
            response = requests.get(url, headers=headers, timeout=10)
            hospitals_raw = response.json()

            # Mesafe hesapla ve filtrele
            hospital_distances = []
            for hospital in hospitals_raw:
                try:
                    lat = float(hospital['lat'])
                    lng = float(hospital['lon'])
                    distance = haversine_distance(user_lat, user_lng, lat, lng)
                    if distance <= max_distance:
                        # Cildiye bölümü anahtar kelime araması
                        name = hospital.get('display_name', '').lower()
                        has_dermatology = any(
                            kw in name for kw in ['cildiye', 'dermatoloji', 'dermatology']
                        )
                        hospital_distances.append({
                            'hospital': hospital,
                            'distance': distance,
                            'has_dermatology': has_dermatology
                        })
                except Exception:
                    continue

            # Mesafeye göre sırala
            hospital_distances.sort(key=lambda x: x['distance'])
            nearby_hospitals = hospital_distances[:limit]

            # JSON formatında döndür
            result = []
            for item in nearby_hospitals:
                hospital = item['hospital']
                result.append({
                    'name': hospital.get('display_name'),
                    'address': hospital.get('display_name'),
                    'latitude': float(hospital['lat']),
                    'longitude': float(hospital['lon']),
                    'has_dermatology': item['has_dermatology'],
                    'distance': round(item['distance'], 2)
                })

            return JsonResponse({
                'success': True,
                'hospitals': result,
                'total_found': len(result)
            })

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            return JsonResponse({
                'success': False,
                'error': f'Geçersiz veri formatı: {str(e)}'
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Sunucu hatası: {str(e)}'
            }, status=500)

    return JsonResponse({
        'success': False,
        'error': 'Sadece POST istekleri kabul edilir'
    }, status=405)

def hospital_search_page(request):
    """
    Hastane arama sayfası
    """
    return render(request, 'hospital_search.html')

# 🆕 CHAT VIEW'LARI - Ne işe yarar:
# 1. chat_page: Chat sayfasını gösterir (kullanıcı chat arayüzünü görür)
# 2. get_chat_messages: Önceki mesajları yükler (sayfa yenilendiğinde eski mesajlar kaybolmaz)

def chat_page(request):
    """
    Chat sayfasını gösterir
    """
    # Kullanıcı için varsayılan oda adı (session ID kullanabiliriz)
    room_name = request.session.session_key or 'general'
    return render(request, 'chat.html', {'room_name': room_name})

def get_chat_messages(request, room_name):
    """
    Belirli bir odadaki mesajları JSON formatında döndürür
    """
    try:
        # Son 50 mesajı al
        messages = ChatMessage.objects.filter(
            room__name=room_name
        ).order_by('-timestamp')[:50]
        
        # Mesajları JSON formatına çevir
        message_list = []
        for message in reversed(messages):  # Eski mesajlar önce
            message_list.append({
                'content': message.content,
                'type': message.message_type,
                'timestamp': message.timestamp.isoformat(),
                'user': message.user.username if message.user else 'Anonymous'
            })
        
        return JsonResponse({
            'success': True,
            'messages': message_list
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
