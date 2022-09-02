from multiprocessing import context
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage             #save the particular file in system
from keras.models import load_model
from keras.preprocessing import image
import json
import numpy as np
import tensorflow as tf
from tensorflow import Graph
# Create your views here.


img_height,img_width = 256,256
with open('models/classes.json','r') as f:
    labelInfo = f.read()    

labelInfo=json.loads(labelInfo)

model = load_model("models/potato_leaf_classify.h5")

# model_graph = Graph()
# with model_graph.as_default():
#     gpuoptions = tf.compat.v1.GPUOptions(allow_growth=True)
#     tf_session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpuoptions))
#     with tf_session.as_default():
#         model = load_model("./models/potato_leaf_classify.h5")


def index(request):
    context = {'a':1}
    return render(request,'index.html',context)


def predictImage(request):

    # To store the uploaded file in Database
    # fs=FileSystemStorage()
    # fileObj = request.FILES['filePath']
    # fileName = fs.save(fileObj.name,fileObj)
    # filePathName = fs.url(fileName)

    # To add the file path to the testImg
    # testimg = filePathName
    #early_blight 
    #testimg = "D:\\SELF\\ML+django\\DeepLearning_PotatoDisease\\res\\TEST\\Potato___Early_blight\\0ddd62cd-a999-4d58-a8f1-506e1004a595___RS_Early.B 8041.JPG"
    #lateblight
    testimg = "D:\\SELF\\ML+django\\DeepLearning_PotatoDisease\\res\\TEST\\Potato___Late_blight\\01a8cc9f-074a-4866-87c8-bb5a9e3895b4___RS_LB 2968.JPG"
    #healthy
    #testimg = "D:\\SELF\\ML+django\\DeepLearning_PotatoDisease\\res\\TEST\\Potato___healthy\\3c0d6888-c7e1-4cf8-9c25-9a0b8c62ba72___RS_HL 1780.JPG"
  
    #loading the test image for the model
    testimg=tf.keras.utils.load_img(testimg,target_size=(img_height,img_width))

    #image_to_array
    testimg_array= tf.keras.utils.img_to_array(testimg)

    testimg_array=testimg_array/255
    testimg_array=testimg_array.reshape(1,img_height,img_width,3)
    #predi=model.predict(x)
    predicted=0
    #Predicting from model on test image
    predicted=model.predict(testimg_array)
    # with model_graph.as_default():
    #     with tf_session.as_default():
    #         predicted=model.predict(testimg_array)
    print("------------")
    model_summary="printed in terminal"
    #model_summary=model.summary()
    print("------------")
    print(str(np.argmax(predicted)))
    print("------------")

    predictLabel = labelInfo[str(np.argmax(predicted))]
    print("------------")

    # print("this is the file path name:",filePathName)
    # print("this is the label name:",predictLabel)
    #context = {'filePathName':filePathName,"predictLabel":predictLabel}
    context = {"labelInfo":labelInfo,"model_summary":model_summary,"predictLabel":predictLabel}
    return render(request,"index.html",context)
