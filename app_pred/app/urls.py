
from django.urls import path, include
from . import views

app_name='app_pred'
urlpatterns = [
    path('',views.home),
    path('form/',views.form,name='form'),
    path('about/',views.about,name='about'),
    path('check/',views.check,name='check'),
]
