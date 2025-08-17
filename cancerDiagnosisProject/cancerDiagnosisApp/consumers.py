import json
import uuid
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.utils import timezone
from .models import ChatRoom, ChatMessage, UserSession
from django.contrib.auth.models import User

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """
        WebSocket bağlantısı kurulduğunda çalışır
        """
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'chat_{self.room_name}'
        
        # Chat odasını oluştur veya al
        self.room = await self.get_or_create_room()
        
        # Kullanıcı oturumunu oluştur
        self.session_id = str(uuid.uuid4())
        await self.create_user_session()
        
        # Gruba katıl
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Hoş geldin mesajı gönder
        await self.send_welcome_message()
    
    async def disconnect(self, close_code):
        """
        WebSocket bağlantısı kesildiğinde çalışır
        """
        # Oturumu güncelle
        await self.update_session_status(False)
        
        # Gruptan çık
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
    
    async def receive(self, text_data):
        """
        WebSocket'ten mesaj alındığında çalışır
        """
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        message_type = text_data_json.get('type', 'user')
        
        # Mesajı veritabanına kaydet
        await self.save_message(message, message_type)
        
        # Bot yanıtı oluştur (eğer kullanıcı mesajıysa)
        if message_type == 'user':
            bot_response = await self.generate_bot_response(message)
            if bot_response:
                await self.save_message(bot_response, 'bot')
                message = bot_response
                message_type = 'bot'
        
        # Gruba mesaj gönder
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message,
                'message_type': message_type,
                'timestamp': timezone.now().isoformat(),
                'session_id': self.session_id
            }
        )
    
    async def chat_message(self, event):
        """
        Chat mesajını WebSocket'e gönderir
        """
        await self.send(text_data=json.dumps({
            'message': event['message'],
            'type': event['message_type'],
            'timestamp': event['timestamp'],
            'session_id': event['session_id']
        }))
    
    @database_sync_to_async
    def get_or_create_room(self):
        """
        Chat odasını oluşturur veya mevcut olanı alır
        """
        room, created = ChatRoom.objects.get_or_create(
            name=self.room_name,
            defaults={'is_active': True}
        )
        return room
    
    @database_sync_to_async
    def create_user_session(self):
        """
        Kullanıcı oturumu oluşturur
        """
        UserSession.objects.create(
            session_id=self.session_id,
            room=self.room,
            is_active=True
        )
    
    @database_sync_to_async
    def update_session_status(self, is_active):
        """
        Oturum durumunu günceller
        """
        try:
            session = UserSession.objects.get(session_id=self.session_id)
            session.is_active = is_active
            session.save()
        except UserSession.DoesNotExist:
            pass
    
    @database_sync_to_async
    def save_message(self, content, message_type):
        """
        Mesajı veritabanına kaydeder
        """
        ChatMessage.objects.create(
            room=self.room,
            message_type=message_type,
            content=content
        )
    
    async def send_welcome_message(self):
        """
        Hoş geldin mesajı gönderir
        """
        welcome_message = (
            "🏥 **Deri Hastalığı Teşhis Sistemi**\n\n"
            "Merhaba! Ben size deri hastalıkları hakkında yardımcı olabilirim.\n\n"
            "**Yapabileceklerim:**\n"
            "• Deri hastalıkları hakkında bilgi verme\n"
            "• Teşhis sonuçlarınızı yorumlama\n"
            "• Hastane önerileri\n"
            "• Genel sağlık tavsiyeleri\n\n"
            "Nasıl yardımcı olabilirim?"
        )
        
        await self.save_message(welcome_message, 'bot')
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': welcome_message,
                'message_type': 'bot',
                'timestamp': timezone.now().isoformat(),
                'session_id': self.session_id
            }
        )
    
    async def generate_bot_response(self, user_message):
        """
        Kullanıcı mesajına göre bot yanıtı oluşturur
        """
        user_message = user_message.lower()
        
        # Basit keyword-based yanıtlar
        responses = {
            'merhaba': 'Merhaba! Size nasıl yardımcı olabilirim?',
            'selam': 'Selam! Deri hastalıkları hakkında sorularınızı yanıtlayabilirim.',
            'teşhis': 'Teşhis sonuçlarınızı yorumlamak için lütfen detayları paylaşın.',
            'hastane': 'Size yakın hastaneleri bulmak için konum bilginizi kullanabilirim.',
            'melanoma': 'Melanoma ciddi bir deri kanseri türüdür. Erken teşhis çok önemlidir.',
            'cilt': 'Cilt sağlığı için güneş koruyucu kullanmayı ve düzenli kontrol yaptırmayı unutmayın.',
            'güneş': 'Güneş ışınları deri kanserine neden olabilir. SPF 30+ güneş koruyucu kullanın.',
            'yardım': 'Size şu konularda yardımcı olabilirim:\n• Deri hastalıkları hakkında bilgi\n• Teşhis yorumlama\n• Hastane önerileri',
        }
        
        # Mesajda keyword ara
        for keyword, response in responses.items():
            if keyword in user_message:
                return response
        
        # Genel yanıt
        return (
            "Anladığım kadarıyla deri sağlığı hakkında bilgi almak istiyorsunuz. "
            "Daha spesifik bir soru sorabilir misiniz? Örneğin:\n"
            "• Belirli bir deri hastalığı hakkında bilgi\n"
            "• Teşhis sonucunuzun yorumlanması\n"
            "• Hastane önerileri"
        ) 