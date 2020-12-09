from django.urls import path
from . import views
app_name = 'para'
urlpatterns = [
        path("para_gen", views.para_gen,name="para_gen"),
        ]