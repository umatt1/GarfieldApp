from django.urls import path

from . import views

urlpatterns = [
path("<str:imagelink>", views.index, name="index"),
]
