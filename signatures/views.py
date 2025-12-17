# from django.shortcuts import render, redirect
# from django.contrib.auth.decorators import login_required
# from .forms import SignatureDocumentForm
# from .models import Signature
# from machine_pipeline.ml_pipeline import SiameseSignatureML
# from PIL import Image
# import io

# def dashbord(request):
#     return render('base.html')

# # Load trained model once
# ml_model = SiameseSignatureML(model_path='v2/signature_model.pth')

# @login_required
# def upload_signature_document(request):
#     user = request.user

#     # Prevent uploading more than 3 reference signatures
#     if Signature.objects.filter(user=user).count() >= 3:
#         return render(request, 'signatures/already_uploaded.html')

#     if request.method == "POST":
#         form = SignatureDocumentForm(request.POST, request.FILES)
#         if form.is_valid():
#             document = request.FILES['document']

#             # Load uploaded image (assumes a scanned paper or image containing 3 signatures)
#             image = Image.open(document).convert("L")  # grayscale

#             # Split into 3 signatures (horizontal split example)
#             width, height = image.size
#             sig_height = height // 3
#             for i in range(3):
#                 sig_img = image.crop((0, i*sig_height, width, (i+1)*sig_height))

#                 # Save each signature in memory
#                 sig_obj = Signature(user=user)
#                 buffer = io.BytesIO()
#                 sig_img.save(buffer, format="PNG")
#                 sig_obj.image.save(f"{user.id}_sig_{i+1}.png", buffer, save=False)

#                 # Extract embedding using trained model
#                 embedding = ml_model.extract_embedding(sig_obj.image.path)
#                 sig_obj.processed_features = embedding.tolist()
#                 sig_obj.save()

#             return redirect('upload_signature_document')
#     else:
#         form = SignatureDocumentForm()

#     return render(request, 'signatures/upload_document.html', {'form': form})


from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import SignatureDocumentForm
from .models import Signature
from machine_pipeline.ml_pipeline import SiameseSignatureML
from PIL import Image
import io

# Dashboard view
def dashboard(request):
    return render(request, 'signatures/base.html')  # make sure this path matches your template

# Load the smaller test model for quick testing
ml_model = SiameseSignatureML(model_path='v2/signature_model_test.pth')

@login_required
def upload_signature_document(request):
    user = request.user

    # Prevent uploading more than 3 reference signatures
    if Signature.objects.filter(user=user).count() >= 3:
        return render(request, 'signatures/already_uploaded.html')

    if request.method == "POST":
        form = SignatureDocumentForm(request.POST, request.FILES)
        if form.is_valid():
            document = request.FILES['document']

            # Load uploaded image as grayscale
            image = Image.open(document).convert("L")

            # Split into 3 horizontal signatures
            width, height = image.size
            sig_height = height // 3
            for i in range(3):
                sig_img = image.crop((0, i*sig_height, width, (i+1)*sig_height))

                # Save each signature to in-memory buffer
                sig_obj = Signature(user=user)
                buffer = io.BytesIO()
                sig_img.save(buffer, format="PNG")
                sig_obj.image.save(f"{user.id}_sig_{i+1}.png", buffer, save=False)

                # Extract embedding using the test model
                embedding = ml_model.extract_embedding(sig_obj.image.path)
                sig_obj.processed_features = embedding.tolist()
                sig_obj.save()

            return redirect('upload_signature_document')
    else:
        form = SignatureDocumentForm()

    return render(request, 'signatures/upload_document.html', {'form': form})

