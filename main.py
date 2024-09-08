import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("\mloading dependencies...")

import numpy as np
import tensorflow as tf

from random import randrange
from time import perf_counter
from PIL import Image

print("loaded!")


model_name = "BCM_LiteTurbo4.h5"

undetermined_range = 15

def getFileExtenstion(filename):
    file_name_parts = filename.split(".")
    return file_name_parts[len(file_name_parts)-1]

def getCancerousAmount(predictions):
    cancerous = 0
    non_cancerous = 0
    
    undetermined = 0

    for prediction in predictions:
        if abs(prediction[0]) < undetermined_range:
            undetermined = undetermined + 1
        elif prediction[0] > undetermined_range:
            non_cancerous = non_cancerous + 1
        elif prediction[0] < -undetermined_range:
            cancerous = cancerous + 1

    rate = 0

    if (cancerous + non_cancerous) != 0:
        rate = (cancerous / (cancerous + non_cancerous)) * 100

    return rate, undetermined, cancerous, non_cancerous 

def getCancerousTag(prediction):
    if abs(prediction[0]) < undetermined_range:
        return "undetermined"
    elif prediction[0] > undetermined_range:
        return "non_cancerous"
    elif prediction[0] < -undetermined_range:
        return "cancerous"

invalid_images = 0
reso = (256, 256)

images_to_classify_dir = "images_to_classify"
amount_of_images = len(os.listdir(images_to_classify_dir))

images_list = []
image_list_names = []

model = tf.keras.models.load_model(model_name)
print(model.summary())

if amount_of_images == 0:
    print(f"0 images are in the |{images_to_classify_dir}| folder. Closing program.")
    sys.exit()

print("Preproccessing images.")

print("\n---------------------------------------\n")

start_create_image = perf_counter()

for count, filename in enumerate(os.listdir(images_to_classify_dir)):
    file_path = 0
    image = 0

    try:
        file_path = os.path.join(images_to_classify_dir, filename)
        image = Image.open(file_path)
    except:
        invalid_images += 1
        print(f"{filename} invalid. [{amount_of_images-count} images remaining]")
        continue
    
    if getFileExtenstion(filename) != "jpg":
        if getFileExtenstion(filename) != "png":
           if getFileExtenstion(filename) != "jpeg": 
                invalid_images += 1
                print(f"{filename} invalid. (file-type invalid) [{amount_of_images-count} images remaining]")
                continue       
    
    if image.size[0] != reso[0] or image.size[1] != reso[1] or image.mode != "RGB":
        invalid_images += 1
        print(f"{filename} invalid. (resultion different or color mode non-RGB) [{amount_of_images-count} images remaining]")
        continue
    
    image_list = []

    for x in range(image.size[0]):
        row = []
        
        for y in range(image.size[1]):
            row.append(image.getpixel((x,y))[0])

        image_list.append(row)
    
    image_list_names.append(filename)
    images_list.append(image_list)

    print(f"{filename} formatted. [{amount_of_images- (count + 1)} images remaining]")

print("\n---------------------------------------\n")

if len(images_list) == 0:
    print(f"all files within the |{images_to_classify_dir}| folder are invalid. Ending Operation.")

print("Running Network on Images...")

p_start_time = perf_counter()
predictions = model.predict(images_list, verbose=2)       
prediction_time = round((perf_counter() - p_start_time) * 1000) / 1000 

print("\nComplete!\n")

print(f"Prediction time: {round(prediction_time, 2)} || Images: {len(images_list)} || Time-Per-Image: {round(prediction_time / len(images_list), 2)}")

print("\n---------------------------------------\n")

report_number = randrange(0, 10000)

print(f"Saving data as Report #{report_number}...\n")

with open(f"report #{report_number}.txt", "a") as f:        
    rate, undetermined, cancerous, non_cancerous = getCancerousAmount(predictions)
    
    f.write(f"Brain Cancer Imaging Report\n\nModel: {model_name}\n\n")      
    f.write(f"Total Prediction Time: {round(prediction_time, 2)}\nTime-Per-Image: {round(prediction_time / len(images_list), 2)}\n\nImages: {len(images_list)}\nDefined Rate: {round(((len(images_list) - undetermined) / len(images_list) * 100), 2)}%\n\n")     
    f.write(f"Cancerous Rate: {round(rate, 2)}%\nNon Cancerous Rate: {round(100 - rate, 2)}%\n\ncancerous: {cancerous}\nnon_cancerous: {non_cancerous}\nundetermined: {undetermined}\nundetermined_range: {undetermined_range}\n\n")     
    
    f.write(f"\n\nRaw Data:\n\n")      

    for prediction, filename in zip(predictions, image_list_names):
        save = f"{filename} || Raw output: {prediction} || Tag: {getCancerousTag(prediction)}\n"
        f.write(save)      
    
print(f"Saved!\n")

create_image_time = round((perf_counter() - start_create_image) * 1000) / 1000 

print(f"Operation took {create_image_time} second(s) to complete!")