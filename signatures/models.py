# from django.db import models
# import uuid
# from django.conf import settings

# class Signature(models.Model):
#     id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
#     user = models.ForeignKey(
#         settings.AUTH_USER_MODEL,  # Links to your User model
#         on_delete=models.CASCADE,
#         related_name='signatures'
#     )
#     image = models.ImageField(upload_to='signatures/')
#     features = models.JSONField(blank=True, null=True)  # ML embedding
#     created_at = models.DateTimeField(auto_now_add=True)

#     def __str__(self):
#         return f"Signature {self.id} for {self.user.email}"
