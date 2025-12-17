import io
import numpy as np
from PIL import Image
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.conf import settings

from .forms import SignatureRegistrationForm
from .models import Signature
from machine_pipeline.ml_pipeline import SiameseSignatureML


# 1. INITIALIZATION

# MODEL_PATH = settings.BASE_DIR / 'machine_pipeline/v2/signature_model.pth'
MODEL_PATH = settings.BASE_DIR / 'machine_pipeline' / 'models' / 'signature_model.pth'
ml_model = SiameseSignatureML(model_path=str(MODEL_PATH))


# 2. DASHBOARD

@login_required
def dashboard(request):
    # # Fetch the user's reference signatures to show on the dashboard
    # user_refs = Signature.objects.filter(user=request.user)
    return render(request, 'base/base.html')


# 3. REGISTRATION (The 3-Signature Upload)

@login_required
def upload_reference_signatures(request):
    """
    Handles 3 separate document uploads. 
    Each file is treated as a unique master sample for the Siamese network.
    """
    if request.method == "POST":
        form = SignatureRegistrationForm(request.POST, request.FILES)
        
        if form.is_valid():
            # Collect the cleaned files from the form
            files = [
                form.cleaned_data['doc1'],
                form.cleaned_data['doc2'],
                form.cleaned_data['doc3']
            ]

            # Clear existing identity for this user to ensure a fresh training set
            Signature.objects.filter(user=request.user).delete()

            for i, uploaded_file in enumerate(files):
                # 1. ML Step: The processor identifies the ink 'blob' and 
                # extracts the 128-D vector fingerprint.
                embedding = ml_model.extract_embedding(uploaded_file)
                
                # 2. Prepare the model instance
                sig_obj = Signature(user=request.user)
                
                # Create a standardized filename: user_1_master_1.jpg
                ext = uploaded_file.name.split('.')[-1]
                filename = f"user_{request.user.id}_master_{i+1}.{ext}"
                
                # 3. Save file and embedding
                sig_obj.image.save(filename, uploaded_file, save=False)
                sig_obj.processed_features = embedding.tolist()
                sig_obj.save()

            return redirect('my_signatures')
    else:
        form = SignatureRegistrationForm()

    return render(request, 'signature/upload_signature.html', {'form': form})


# 4. VERIFICATION (The Match/No-Match Logic)

@login_required
def verify_signature_view(request):
    """
    Standard Template View: Compares a new upload and renders a result page.
    """
    if request.method == "POST" and request.FILES.get('signature_to_verify'):
        uploaded_file = request.FILES['signature_to_verify']
        
        # 1. Extract embedding from the NEW signature
        # ml_model.extract_embedding handles the binarization/preprocessing
        new_embedding = ml_model.extract_embedding(uploaded_file)

        # 2. Fetch all stored reference signatures for this user
        references = Signature.objects.filter(user=request.user)
        
        if not references.exists():
            return render(request, 'signature/error.html', {
                'error_message': 'You have not registered any signatures yet. Please go to registration first.'
            })

        # 3. Compare new signature against ALL 3 stored references
        distances = []
        for ref in references:
            stored_vector = np.array(ref.processed_features)
            # Calculate Euclidean distance using our ML helper
            dist = ml_model.calculate_distance(stored_vector, new_embedding)
            distances.append(float(dist))

        # 4. Decision Logic (Take the best match out of the three)
        min_dist = min(distances)
        
        # THRESHOLD SETTING: 0.5 (Balanced), 0.4 (Strict Security)
        is_genuine = min_dist < 0.5 
        
        # Calculate a friendly score for the UI
        confidence = round((1 - min_dist) * 100, 2)
        if confidence > 100: confidence = 100.0
        if confidence < 0: confidence = 0.0

        # 5. RENDER the result to a template
        context = {
            'is_genuine': is_genuine,
            'confidence': confidence,
            'distance': round(min_dist, 4),
            'references': references  # Pass refs so user can see them side-by-side
        }
        return render(request, 'signature/verify_result.html', context)

    # If it's a GET request, just show the upload form
    return render(request, 'signature/verify_signature.html')


@login_required
def my_signature_view(request):
    # Fetch the 3 reference signatures for the dashboard
    references = Signature.objects.filter(user=request.user)
    
    return render(request, 'signature/my_signature.html', {
        'references': references
    })