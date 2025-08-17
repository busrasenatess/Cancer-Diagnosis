from django.contrib import admin
from .models import CancerDiagnosis, DatasetInfo, ChatRoom, ChatMessage, UserSession

@admin.register(CancerDiagnosis)
class CancerDiagnosisAdmin(admin.ModelAdmin):
    list_display = ['image_name', 'diagnosis', 'confidence', 'uploaded_at']
    list_filter = ['diagnosis', 'uploaded_at']
    search_fields = ['image_name', 'diagnosis']
    readonly_fields = ['uploaded_at']
    ordering = ['-uploaded_at']

@admin.register(DatasetInfo)
class DatasetInfoAdmin(admin.ModelAdmin):
    list_display = ['name', 'total_images', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at']
    ordering = ['-created_at']

# 🆕 CHAT ADMIN - Ne işe yarar:
# 1. ChatRoomAdmin: Chat odalarını yönetir (aktif/pasif yapma, silme)
# 2. ChatMessageAdmin: Mesajları görüntüler ve filtreler
# 3. UserSessionAdmin: Kullanıcı oturumlarını takip eder

@admin.register(ChatRoom)
class ChatRoomAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_at', 'is_active', 'message_count']
    list_filter = ['is_active', 'created_at']
    search_fields = ['name']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
    
    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = 'Mesaj Sayısı'

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['room', 'message_type', 'content_preview', 'timestamp', 'user']
    list_filter = ['message_type', 'timestamp', 'room']
    search_fields = ['content', 'room__name']
    readonly_fields = ['timestamp']
    ordering = ['-timestamp']
    
    def content_preview(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    content_preview.short_description = 'Mesaj Önizleme'

@admin.register(UserSession)
class UserSessionAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'user', 'room', 'created_at', 'last_activity', 'is_active']
    list_filter = ['is_active', 'created_at', 'last_activity']
    search_fields = ['session_id', 'user__username', 'room__name']
    readonly_fields = ['created_at', 'last_activity']
    ordering = ['-last_activity']
