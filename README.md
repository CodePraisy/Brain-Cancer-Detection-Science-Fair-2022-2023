# **PROTOTYPE/PROOF OF CONCEPT**


## Objective
The primary goal of this project is to create a prototype or proof of concept for an early-stage brain cancer detection system using machine learning. This model aims to assist in identifying cancerous growths or abnormalities in brain scans, which can lead to faster and more accurate diagnoses.

## Scope
This project serves as a prototype and proof of concept. It is not intended for real-world diagnostic use but instead aims to showcase the potential of machine learning in healthcare, specifically in the detection of brain cancer through medical imaging.

This upload is **not** different then the version from early 2023, it is just being uploaded later.

## Components
### Data Source:

A dataset of brain scans (MRI/CT scans) that includes both cancerous and non-cancerous images.
For the prototype, this kaggle dataset was used: https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri

### Machine Learning Model:

The model used for this project is based on different variations of the well known Convolutional Neural Network (CNN), which has proven effective for image classification and feature extraction in medical images.
To access one of the models trained from this program, download them from here: https://drive.google.com/drive/folders/1AYPeCEJi_rwd7_c1HENesPt1AD2VVO9X?usp=sharing

### Programming Language & Libraries:

Language: **Python**

Libraries:
```
TensorFlow/Keras/sklearn (for model building and training)
NumPy (for data handling and preprocessing)
OpenCV/Pillow (for image manipulation and augmentation)
Pickle (for saving preproccessed images for training)
Matplotlib, Seaborn (for data visualization)
```
## Get Started

1. Classify your train images of choice and split them into two batches, cancer and non cancer (the kaggle images should already be sorted, but if you'd like to use your own make sure to complete this step first)
2. Bake the images using the bake_dataset.py file which converts them into pickle files, make sure to change the file names the bake_dataset python file is looking for.
3. Change the code in the network.py file to use the generated pickle files and pick one of the neural architecture or make your own and train the model.
4. Input the created model (or one of the imported models) into the models folder and test the accuracy. (make sure to put the images you'd like to test the model on in the images_to_classify folder)
5. Once you train the model to your satifcation, you can use the model in a project or file elsewhere.
