from django.urls import path

from . import views

urlpatterns = [
path("<path:imagelink>", views.index, name="index"),
]
