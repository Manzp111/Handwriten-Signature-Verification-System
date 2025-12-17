from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .forms import UserRegistrationForm, UserLoginForm
from django.contrib.auth.hashers import check_password
from .models import User


# User Registration

def register(request):
    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserRegistrationForm()
    return render(request, "users/register.html", {"form": form})


# User Login (Corrected Logic)

def user_login(request):
    # Initialize form once at the top to prevent UnboundLocalError
    form = UserLoginForm()

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
                
                # Check if the user has any registered signatures
                # This uses the related_name="signatures" from your Signature model
                if not user.signatures.exists():
                    return redirect("upload_reference_signatures")
                else:
                    return redirect("dashboard")
            else:
                form.add_error(None, "Authentication failed. Please contact support.")

    # This return handles initial GET requests AND POST requests that failed validation
    return render(request, "users/login.html", {"form": form})


# User Logout

def user_logout(request):
    logout(request)
    return redirect("login")