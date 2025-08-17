from django.urls import path
from . import views

urlpatterns = [
    path('', views.HomePage, name='home'),
    path('hospitals/', views.hospital_search_page, name='hospital_search'),
    path('api/nearby-hospitals/', views.get_nearby_hospitals, name='get_nearby_hospitals'),
    path('chat/', views.chat_page, name='chat'),
    path('api/chat/messages/<str:room_name>/', views.get_chat_messages, name='get_chat_messages'),
]