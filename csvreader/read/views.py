from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.forms import AuthenticationForm
from .forms import CustomSignupForm

# Create your views here.
def index(request):
    return render(request, 'index.html')


# 1. SIGNUP LOGIC
def signup_view(request):
    if request.method == 'POST':
        # User submitted data. Validate it.
        form = CustomSignupForm(request.POST)
        if form.is_valid():
            user = form.save() # Saves user to DB and hashes password
            login(request, user) # Auto-login after signup
            return redirect('index') # Redirect to the main dashboard
    else:
        # User just opened the page. Show empty form.
        form = CustomSignupForm()
    
    return render(request, 'registration/signup.html', {'form': form})

# 2. LOGIN LOGIC
def login_view(request):
    if request.method == 'POST':
        # 'AuthenticationForm' checks username/password against the DB
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            # Get the user object
            user = form.get_user()
            # This creates the Session ID cookie
            login(request, user)
            # 'next' handles where to go if they were forced to login
            if 'next' in request.POST:
                return redirect(request.POST.get('next'))
            return redirect('index')
    else:
        form = AuthenticationForm()
    
    return render(request, 'registration/login.html', {'form': form})

# 3. LOGOUT LOGIC
def logout_view(request):
    if request.method == 'POST':
        logout(request) # Deletes the session cookie
        return redirect('login')
