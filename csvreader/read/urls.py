from django.urls import path
from . import views

urlpatterns = [
    # Auth Routes
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # App Routes
    path('', views.upload_view, name='upload'),  # Homepage is Upload
    path('select/<int:dataset_id>/', views.select_target_view, name='select_target'),
    path('result/', views.train_model_view, name='result'), # (Internal use mostly)
    path('predict/<int:dataset_id>/', views.predict_view, name='predict'),
    path('predict-trend/', views.predict_future_trend, name='predict_future_trend'),
]