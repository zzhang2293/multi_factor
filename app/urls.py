from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views
urlpatterns = [
    path('app/', views.members, name = 'members'),
    path('app/collect_data/', views.collect_data, name='datas'),
    path('app/factors/', views.get_factors, name="factor"),
    path('app/download_zip/', views.get_csv_output, name="csv_data"),
    path('app/download_factor_zip/', views.get_factor_csv_output, name="csv_factor_data")

]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)