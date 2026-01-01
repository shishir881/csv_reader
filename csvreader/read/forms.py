from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .models import Dataset

# Tailwind Styling for Inputs
INPUT_CLASSES = "w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"

# --- 1. CSV UPLOAD FORM ---
class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={'class': INPUT_CLASSES, 'accept': '.csv'})
        }

class PredictionForm(forms.Form):
    file = forms.FileField(
        label="Upload New Data (CSV)",
        widget=forms.FileInput(attrs={'class': INPUT_CLASSES, 'accept': '.csv'})
    )

# --- 2. AUTH FORMS (Existing updated) ---
class CustomSignupForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'class': INPUT_CLASSES}))
    class Meta:
        model = User
        fields = ['username', 'email']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields:
            self.fields[field].widget.attrs.update({'class': INPUT_CLASSES})

class CustomLoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields:
            self.fields[field].widget.attrs.update({'class': INPUT_CLASSES})