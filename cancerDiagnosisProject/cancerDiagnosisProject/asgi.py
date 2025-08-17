"""
ASGI config for cancerDiagnosisProject project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cancerDiagnosisProject.settings')

def get_application():
    from cancerDiagnosisApp.routing import get_websocket_urlpatterns
    return ProtocolTypeRouter({
        "http": get_asgi_application(),
        "websocket": AuthMiddlewareStack(
            URLRouter(
                get_websocket_urlpatterns()
            )
        ),
    })

application = get_application()
