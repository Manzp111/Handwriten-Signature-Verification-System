from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager

# User Manager

class UserManager(BaseUserManager):
    def create_user(self, id, email, password=None, full_name=None, account_number=None):
        if not id:
            raise ValueError("Users must have an integer ID")
        if not email:
            raise ValueError("Users must have an email address")
        if not account_number:
            raise ValueError("Users must have an account number")
        
        email = self.normalize_email(email)
        user = self.model(id=id, email=email, full_name=full_name, account_number=account_number)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, id, email, password, full_name=None, account_number=None):
        user = self.create_user(id=id, email=email, password=password, full_name=full_name, account_number=account_number)
        user.is_admin = True
        user.is_superuser = True
        user.save(using=self._db)
        return user

# BankUser Model
class User(AbstractBaseUser):
    id = models.IntegerField(primary_key=True)  # manually provided integer ID
    full_name = models.CharField(max_length=200)
    email = models.EmailField(unique=True)
    account_number = models.CharField(max_length=20, unique=True)
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['id', 'full_name', 'account_number']

    def __str__(self):
        return f"{self.full_name} ({self.email})"

    # Permissions for admin
    def has_perm(self, perm, obj=None):
        return self.is_admin

    def has_module_perms(self, app_label):
        return self.is_admin

    @property
    def is_staff(self):
        return self.is_admin
