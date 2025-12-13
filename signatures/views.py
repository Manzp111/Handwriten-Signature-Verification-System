from django.shortcuts import render, redirect
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views import View
from .forms import SignatureUploadForm
from .models import Signature
from .utils import preprocess_image, SiameseCNN, verify_signature
import torch

# ----------------------------
# Instantiate ML model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseCNN().to(device)
model.eval()

# ----------------------------
# Upload Signature
# ----------------------------
class SignatureUploadView(LoginRequiredMixin, View):
    def get(self, request):
        form = SignatureUploadForm()
        return render(request, 'upload_signature.html', {'form': form})

    def post(self, request):
        form = SignatureUploadForm(request.POST, request.FILES)
        if form.is_valid():
            signature = form.save(commit=False)
            signature.user = request.user

            # Extract features
            input_tensor = preprocess_image(request.FILES['image']).to(device).flatten()
            signature.features = input_tensor.tolist()
            signature.save()
            
            return render(request, 'upload_signature.html', {
                'form': SignatureUploadForm(),
                'message': 'Signature uploaded successfully!'
            })
        return render(request, 'upload_signature.html', {'form': form})


# ----------------------------
# Verify Signature
# ----------------------------
class SignatureVerifyView(LoginRequiredMixin, View):
    def get(self, request):
        return render(request, 'verify_signature.html')

    def post(self, request):
        file = request.FILES.get('signature')
        if not file:
            return render(request, 'verify_signature.html', {'error': 'No file provided'})
        
        result = verify_signature(request.user, file, model=model, device=device)
        return render(request, 'verify_signature.html', {'result': result})


# ----------------------------
# Staff: List all Signatures
# ----------------------------
class SignatureListView(LoginRequiredMixin, UserPassesTestMixin, View):
    def test_func(self):
        return self.request.user.is_staff

    def get(self, request):
        signatures = Signature.objects.all()
        return render(request, 'signature_list.html', {'signatures': signatures})
