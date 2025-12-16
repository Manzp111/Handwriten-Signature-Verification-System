from django.urls import path
from .views import dashbord

urlpatterns = [
    path('dashbord/', dashbord, name='dashhbord'),
   
]
