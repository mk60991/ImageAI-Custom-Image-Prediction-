# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:35:54 2018

@author: hp
"""

from imageai.Prediction.Custom import CustomImagePrediction
import os
execution_path = os.getcwd()


prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "C:\\Users\\hp\\Downloads\\imageAI custom image prediction\\resnet_model_ex-020_acc-0.651714.h5"))
prediction.setJsonPath(os.path.join(execution_path, "C:\\Users\\hp\\Downloads\\imageAI custom image prediction\\model_class.json"))
prediction.loadModel(num_objects=10)


predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "C:\\Users\\hp\\Downloads\\imageAI custom image prediction\\input images\\4.jpg"), result_count=5)


for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction + " : " + eachProbability)