from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .forms import UserRegistrationForm, UserLoginForm
from django.contrib.auth.hashers import check_password
from .models import User


# ----------------------------
# User Registration
# ----------------------------
def register(request):
    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()  # password + auto fields handled in form & manager
            return redirect('login')
    else:
        form = UserRegistrationForm()

    return render(request, "users/register.html", {"form": form})


# ----------------------------
# User Login
# ----------------------------
def user_login(request):
    if request.method == "POST":
        form = UserLoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"]
            password = form.cleaned_data["password"]

            # 1. Check if email exists
            try:
                user_obj = User.objects.get(email=email)
            except User.DoesNotExist:
                form.add_error("email", "Email does not exist")
                return render(request, "users/login.html", {"form": form})

            # 2. Check password
            if not check_password(password, user_obj.password):
                form.add_error("password", "Incorrect password")
                return render(request, "users/login.html", {"form": form})

            # 3. Authenticate & login
            user = authenticate(request, email=email, password=password)
            if user is not None:
                login(request, user)
                return redirect("dashboard")

    else:
        form = UserLoginForm()

    return render(request, "users/login.html", {"form": form})


# ----------------------------
# User Logout
# ----------------------------
def user_logout(request):
    logout(request)
    return redirect("login")
