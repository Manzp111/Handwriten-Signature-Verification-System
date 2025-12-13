from django.shortcuts import render, redirect
from .forms import UserRegistrationForm
from django.contrib.auth import authenticate, login, logout
from .forms import UserLoginForm

def register(request):
    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            # Hash the password
            user.set_password(form.cleaned_data['password'])
            user.save()
            return redirect('login')  # Redirect to login page
    else:
        form = UserRegistrationForm()

    return render(request, "users/register.html", {"form": form})



def user_login(request):
    if request.method == "POST":
        form = UserLoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user = authenticate(request, email=email, password=password)
            if user is not None:
                login(request, user)
                return redirect('dashboard')  # Redirect to a dashboard or signature upload page
            else:
                form.add_error(None, "Invalid email or password")
    else:
        form = UserLoginForm()
    return render(request, "users/login.html", {"form": form})



def user_logout(request):
    logout(request)
    return redirect('login')

