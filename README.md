# ImageAI-Custom-Image-Prediction-

ImageAI : Custom Image Prediction 
An AI Commons project https://commons.specpal.science



ImageAI provides 4 different algorithms and model types to perform custom image prediction using your custom models. You will be able to use your model trained with ImageAI and the corresponding model_class JSON file to predict custom objects that you have trained the model on. In this example, we will be using the model trained for 20 experiments on IdenProf, a dataset of uniformed professionals and achieved 65.17% accuracy on the test dataset. Download the ResNet model of the model and JSON files in links below: 

- ResNet (Size = 90.4 mb) 
https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0.1/resnet_model_ex-020_acc-0.651714.h5

- IdenProf model_class.json file 
https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0.1/model_class.json


Great! Once you have downloaded this model file and the JSON file, start a new python project, and then copy the model file and the JSON file to your project folder where your python files (.py files) will be . Download the image below, or take any image on your computer that include any of the following professionals(Chef, Doctor, Engineer, Farmer, Fireman, Judge, Mechanic, Pilot, Police and Waiter) and copy it to your python project's folder. Then create a python file and give it a name; an example is FirstCustomPrediction.py. Then write the code below into the python file: 

FirstCustomPrediction.py
from imageai.Prediction.Custom import CustomImagePrediction
import os
execution_path = os.getcwd()


prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "resnet_model_ex-020_acc-0.651714.h5"))
prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
prediction.loadModel(num_objects=10)


predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "4.jpg"), result_count=5)


for eachPrediction, eachProbability in zip(predictions, probabilities):
print(eachPrediction + " : " + eachProbability)

Sample Result: 


mechanic : 76.82620286941528
chef : 10.106072574853897
waiter : 4.036874696612358
police : 2.6663416996598244
pilot : 2.239348366856575

The code above works as follows: 

from imageai.Prediction.Custom import CustomImagePrediction
import os

The code above imports the ImageAI library for custom image prediction and the python os class. 
execution_path = os.getcwd()

The above line obtains the path to the folder that contains your python file (in this example, your FirstCustomPrediction.py) . 

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "resnet_model_ex-020_acc-0.651714.h5"))
prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
prediction.loadModel(num_objects=10)
In the lines above, we created and instance of the CustomImagePrediction() class in the first line, then we set the model type of the prediction object to ResNet by caling the .setModelTypeAsResNet() in the second line, we set the model path of the prediction object to the path of the custom model file (resnet_model_ex-020_acc-0.651714.h5) we copied to the python file folder in the third line, we set the path to the model_class.json of the model, we load the model and parse the number of objected that can be predicted in the model.

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "4.jpg"), result_count=5)
In the above line, we defined 2 variables to be equal to the function called to predict an image, which is the .predictImage() function, into which we parsed the path to our image and also state the number of prediction results we want to have (values from 1 to 10 in this case) parsing result_count=5 . The .predictImage() function will return 2 array objects with the first (predictions) being an array of predictions and the second (percentage_probabilities) being an array of the corresponding percentage probability for each prediction.

for eachPrediction, eachProbability in zip(predictions, probabilities):
print(eachPrediction + " : " + eachProbability)
The above line obtains each object in the predictions array, and also obtains the corresponding percentage probability from the percentage_probabilities, and finally prints the result of both to console.



CustomImagePrediction class also supports the multiple predictions, input types and prediction speeds that are contained in the ImagePrediction class. Follow this link to see all the details.

