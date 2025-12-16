# signatures/forms.py

from django import forms

class SignatureDocumentForm(forms.Form):
    """
    Form for uploading a single document that contains 3 genuine signatures.
    """
    document = forms.ImageField(
        label="Upload Signature Document",
        help_text="Upload a scanned image containing 3 signatures (PNG, JPG, JPEG).",
        widget=forms.ClearableFileInput(attrs={'accept': 'image/*'})
    )
