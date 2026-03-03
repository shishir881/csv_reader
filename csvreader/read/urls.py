from django.urls import path
from . import views

urlpatterns = [
    # Landing page (public)
    path('', views.landing_view, name='landing'),

    # Auth
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Dashboard & upload
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('upload/', views.upload_view, name='upload'),

    # ML workflow
    path('dataset/<int:dataset_id>/select/', views.select_target_view, name='select_target'),
    path('dataset/<int:dataset_id>/result/', views.train_model_view, name='result'),
    path('dataset/<int:dataset_id>/predict/', views.predict_view, name='predict'),
    path('dataset/<int:dataset_id>/trend/', views.predict_future_trend, name='predict_future_trend'),

    # Delete dataset
    path('dataset/<int:dataset_id>/delete/', views.delete_dataset_view, name='delete_dataset'),
]