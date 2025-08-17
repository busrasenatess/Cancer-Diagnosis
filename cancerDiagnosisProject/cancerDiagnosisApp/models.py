from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class CancerDiagnosis(models.Model):
    """
    Kanser teşhis sonuçlarını saklayan model
    """
    DIAGNOSIS_CHOICES = [
        ('akiec', 'Actinic Keratoses ve Intraepithelial Carcinoma'),
        ('bcc', 'Basal Cell Carcinoma'),
        ('bkl', 'Benign Keratosis Benzeri Lezyonlar'),
        ('df', 'Dermatofibroma'),
        ('mel', 'Melanoma'),
        ('nv', 'Melanocytic Nevi'),
        ('vasc', 'Vasküler Lezyonlar'),
    ]
    
    image_name = models.CharField(max_length=255)
    diagnosis = models.CharField(max_length=50, choices=DIAGNOSIS_CHOICES)
    confidence = models.FloatField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.image_name} - {self.diagnosis} ({self.confidence:.1f}%)"
    
    class Meta:
        verbose_name = "Kanser Teşhisi"
        verbose_name_plural = "Kanser Teşhisleri"

class DatasetInfo(models.Model):
    """
    Dataset hakkında bilgi saklayan model
    """
    name = models.CharField(max_length=100)
    description = models.TextField()
    class_names = models.JSONField()  # Sınıf isimlerini JSON olarak sakla
    total_images = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} - {self.total_images} görsel"
    
    class Meta:
        verbose_name = "Dataset Bilgisi"
        verbose_name_plural = "Dataset Bilgileri"

# Chat Modelleri
class ChatRoom(models.Model):
    """
    Chat odası modeli
    """
    name = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"Chat Room: {self.name}"
    
    class Meta:
        verbose_name = "Chat Odası"
        verbose_name_plural = "Chat Odaları"

class ChatMessage(models.Model):
    """
    Chat mesajı modeli
    """
    MESSAGE_TYPES = [
        ('user', 'Kullanıcı'),
        ('bot', 'Bot'),
        ('system', 'Sistem'),
    ]
    
    room = models.ForeignKey(ChatRoom, on_delete=models.CASCADE, related_name='messages')
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES, default='user')
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.room.name} - {self.user.username if self.user else 'Anonymous'} - {self.timestamp}"
    
    class Meta:
        verbose_name = "Chat Mesajı"
        verbose_name_plural = "Chat Mesajları"
        ordering = ['timestamp']

class UserSession(models.Model):
    """
    Kullanıcı oturum bilgileri
    """
    session_id = models.CharField(max_length=255, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    room = models.ForeignKey(ChatRoom, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"Session: {self.session_id} - {self.user.username if self.user else 'Anonymous'}"
    
    class Meta:
        verbose_name = "Kullanıcı Oturumu"
        verbose_name_plural = "Kullanıcı Oturumları"
