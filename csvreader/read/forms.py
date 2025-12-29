from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

# 1. SIGNUP FORM (Customized to add Email)
class CustomSignupForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email'] # Password is added automatically by UserCreationForm