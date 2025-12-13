from django import forms
from .models import Signature

class SignatureUploadForm(forms.ModelForm):
    class Meta:
        model = Signature
        fields = ['image']
