from django import forms
from .models import Signature


class SignatureRegistrationForm(forms.Form):
    doc1 = forms.ImageField(
        label="Master Sample 1",
        widget=forms.FileInput(attrs={'class': 'hidden signature-input', 'data-id': '1'})
    )
    doc2 = forms.ImageField(
        label="Master Sample 2",
        widget=forms.FileInput(attrs={'class': 'hidden signature-input', 'data-id': '2'})
    )
    doc3 = forms.ImageField(
        label="Master Sample 3",
        widget=forms.FileInput(attrs={'class': 'hidden signature-input', 'data-id': '3'})
    )

class VerifySignatureForm(forms.Form):    
    signature_to_verify = forms.ImageField(label="Upload Signature for Verification")