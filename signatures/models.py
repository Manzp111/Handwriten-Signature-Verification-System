from django.db import models
from Users.models import User  # your custom User model


class Signature(models.Model):
    """
    Model to store genuine reference signatures for each user.
    Test signatures are never stored in DB.
    """
    id = models.AutoField(primary_key=True)
    
    # Link signature to the user
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="signatures"
    )
    
    # The actual signature image
    image = models.ImageField(
        upload_to="signatures/",
        help_text="Upload genuine handwritten signature image"
    )
    
    # Optional: store precomputed ML features (HOG/CNN embeddings)
    processed_features = models.JSONField(
        blank=True,
        null=True,
        help_text="Optional: store features for faster ML verification"
    )
    
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Signature"
        verbose_name_plural = "Signatures"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.full_name} - Signature {self.id}"
