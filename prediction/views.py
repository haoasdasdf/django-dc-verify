from django.shortcuts import render
from .forms import ImageUploadForm
import random
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions

from io import BytesIO
import os

# Create your views here.
def dc(request):
    if request.method == 'GET':
        # GETリクエストによるアクセス時の処理を記述
        form = ImageUploadForm()
        return render(request, 'dc.html', {'form': form})
    if request.method == 'POST':
        # POSTリクエストによるアクセス時の処理を記述
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            # 画像ファイル（img_file）の前処理を追加
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(256, 256))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = img_array.reshape((1, 256, 256, 3))
            img_array = img_array/255
            # 判定結果のロジックを追加
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'model.h5')
            model = load_model(model_path)
            result = model.predict(img_array)
            if result[0][0] > result[0][1]:
                prediction = '猫'
            else:
                prediction = '犬'

            img_data = request.POST.get('img_data')
            return render(request, 'dc.html', {'form': form, 'prediction': prediction, 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'dc.html', {'form': form})
        
# def things(request):
#     if request.method == 'GET':
#         # GETリクエストによるアクセス時の処理を記述
#         form = ImageUploadForm()
#         return render(request, 'things.html', {'form': form})
#     if request.method == 'POST':
#         # POSTリクエストによるアクセス時の処理を記述
#         form = ImageUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             img_file = form.cleaned_data['image']
#             # 画像ファイル（img_file）の前処理を追加
#             img_file = BytesIO(img_file.read())
#             img = load_img(img_file, target_size=(224, 224))
#             img_array = img_to_array(img)
#             img_array = img_array.reshape((1, 224, 224, 3))
#             img_array = preprocess_input(img_array)

#             # 判定結果のロジックを追加

#             model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
#             model = load_model(model_path)
#             result = model.predict(img_array)
#             prediction = decode_predictions(result, top=5)[0]
#             print(prediction)
            
#             img_data = request.POST.get('img_data')
#             return render(request, 'things.html', {'form': form, 'prediction': prediction, 'img_data': img_data})
#         else:
#             form = ImageUploadForm()
#             return render(request, 'things.html', {'form': form})