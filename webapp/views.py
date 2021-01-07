from django.shortcuts import render
from django.core.files.storage import FileSystemStorage  # to store input file

from keras.models import load_model
from keras.preprocessing import image
import json
import numpy as np
from tensorflow import Graph, Session

# Create your views here.

# ml properties
img_height, img_width = 224, 224
with open('./notebook/imagenet_classes.json', 'r') as f:
    labelInfo = f.read()

labelInfo = json.loads(labelInfo)


# model = load_model('./notebook/MobileNetModelImagenet.h5')

# Tensor Tensor("act_softmax/Softmax:0", shape=(?, 1000), dtype=float32) is not an element of this graph.
model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model = load_model('./notebook/MobileNetModelImagenet.h5')


def index(request):
    ctx = {}
    return render(request, 'index.html', context=ctx)


def predictImage(request):
    print(request, request.POST.dict(), request.FILES['filePath'])

    # fetching file fron request and saving
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    # file will be saved to MEDIA_ROOT
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)

    feedImagePath = '.' + filePathName

    # predict image
    img = image.load_img(feedImagePath, target_size=(img_height, img_width))

    x = image.img_to_array(img)
    x = x/255
    x = x.reshape(1, img_height, img_width, 3)

    with model_graph.as_default():
        with tf_session.as_default():
            predi = model.predict(x)

    prob_index = np.argmax(predi[0])

    label = labelInfo[f'{prob_index}'][1]

    ctx = {
        'filePathName': filePathName,
        'predictedLabel': label,
    }
    return render(request, 'index.html', context=ctx)

# def viewDataBase(request):
