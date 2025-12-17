from django.urls import path
from .views import dashboard,upload_reference_signatures,verify_signature_view,my_signature_view

urlpatterns = [
    path('register/', upload_reference_signatures, name='upload_reference_signatures'),
    path('verify/', verify_signature_view, name='verify_signature'),
    path('dashboard/', dashboard, name='dashboard'),
    path('my-signatures/', my_signature_view, name='my_signatures'), 
]
