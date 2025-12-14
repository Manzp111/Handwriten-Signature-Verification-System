from django.db import models, transaction, IntegrityError
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
import uuid


# ----------------------------
# User Manager
# ----------------------------
class UserManager(BaseUserManager):

    def generate_account_number(self):
        """
        Generates a unique incremental 12-digit account number starting with 70
        Example: 700000000001, 700000000002, ...
        """
        last_user = self.model.objects.order_by('id').last()
        if not last_user:
            new_number = 700000000001
        else:
            new_number = int(last_user.account_number) + 1
        return str(new_number)

    def create_user(self, email, password=None, full_name=None):
        if not email:
            raise ValueError("Users must have an email address")
        if not password:
            raise ValueError("Users must have a password")
        if not full_name:
            raise ValueError("Users must have a full name")

        email = self.normalize_email(email)

        # Create user with retry in case of rare race condition
        for _ in range(5):
            try:
                with transaction.atomic():
                    user = self.model(
                        email=email,
                        full_name=full_name,
                        account_number=self.generate_account_number(),
                        uuid=uuid.uuid4()  # unique identifier
                    )
                    user.set_password(password)
                    user.is_active = True
                    user.save(using=self._db)
                return user
            except IntegrityError:
                continue
        raise ValueError("Failed to generate a unique account number after 5 tries")

    def create_superuser(self, email, password, full_name="Admin User"):
        user = self.create_user(email=email, password=password, full_name=full_name)
        user.is_admin = True
        user.is_superuser = True
        user.save(using=self._db)
        return user


# ----------------------------
# User Model
# ----------------------------
class User(AbstractBaseUser):
    id = models.AutoField(primary_key=True)  # auto-incremented primary key
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    full_name = models.CharField(max_length=200)
    email = models.EmailField(unique=True)
    account_number = models.CharField(max_length=12, unique=True)

    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['full_name']

    def __str__(self):
        return f"{self.full_name} ({self.email})"

    # Permissions
    def has_perm(self, perm, obj=None):
        return self.is_admin

    def has_module_perms(self, app_label):
        return self.is_admin

    @property
    def is_staff(self):
        return self.is_admin
