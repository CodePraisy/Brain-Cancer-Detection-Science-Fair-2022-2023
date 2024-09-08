import os
import random
import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from time import perf_counter
from random import randrange
from PIL import Image

undetermined_range = 15

def getModelAccuracy(predictions, anwser_sheet):
    right = 0
    wrong = 0
    
    undetermined = 0

    for prediction, anwser in zip(predictions, anwser_sheet):
        if abs(prediction[0]) < undetermined_range:
            undetermined = undetermined + 1
        elif prediction[0] > undetermined_range and anwser == 1:
            right = right + 1
        elif prediction[0] < -undetermined_range and anwser == 0:
            right = right + 1
        elif prediction[0] > undetermined_range and anwser == 0:
            wrong = wrong + 1
        elif prediction[0] < -undetermined_range and anwser == 1:
            wrong = wrong + 1

    rate = 0

    if (right + wrong) != 0:
        rate = (right / (right + wrong)) * 100

    return rate, undetermined, right, wrong 

def getAccuracyTag(prediction, anwser):
    if abs(prediction[0]) < undetermined_range:
        return "undetermined"
    elif prediction[0] > undetermined_range and anwser == 1:
        return "right"
    elif prediction[0] < -undetermined_range and anwser == 0:
        return "right"
    elif prediction[0] > undetermined_range and anwser == 0:
        return "wrong"
    elif prediction[0] < -undetermined_range and anwser == 1:
        return "wrong"

def getAnwserTag(number):
    if number == 0:
        return "cancer_positive"
    else:
        return "cancer_negitive"

def getFileExtenstion(filename):
    file_name_parts = filename.split(".")
    return file_name_parts[len(file_name_parts)-1].lower()

images_to_classify_dirs = [["cleaned_dataset\\cancer", "cancer"], ["cleaned_dataset_test\\cancer", "cancer"], ["cleaned_dataset\\no_cancer", "not_cancer"], ["cleaned_dataset_test\\no_cancer", "not_cancer"]]
models_folder = "models_to_test"

reso = (256, 256)

invalid_images = 0

models = []
model_names = []
unloaded_models = []

images_list = []
image_list_names = []

# build models 

loading_models_time = perf_counter()

for model_filename in os.listdir(models_folder):
    try:
        model = tf.keras.models.load_model(os.path.join(models_folder, model_filename))
        
        models.append(model)
        model_names.append(os.path.splitext(model_filename)[0])
        
        print(f"{model_filename} loaded!")
    except:
        print(f"{model_filename} couldn't be loaded.")
        unloaded_models.append(os.path.join(models_folder, model_filename))

loading_models_time = (perf_counter() - loading_models_time)

# load_images

loading_images_time = perf_counter()

print("\nLoading images...\n")

for folder_count, directory in enumerate(images_to_classify_dirs):
    folder_count = folder_count + 1
    
    amount_of_images = len(os.listdir(directory[0]))

    for count, filename in enumerate(os.listdir(directory[0])):
        count = count + 1
        
        filepath = 0
        image = 0

        try:
            filepath = os.path.join(directory[0], filename)
            image = Image.open(filepath)
        except:
            invalid_images += 1
            print(f"{filepath} invalid. [{amount_of_images - count} images remaining]")
            continue

        if getFileExtenstion(filename) != "jpg":
            if getFileExtenstion(filename) != "png":
                if getFileExtenstion(filename) != "jpeg": 
                    invalid_images += 1
                    print(f"{filepath} invalid. (file-type invalid) [{amount_of_images - count} images remaining]")
                    continue 

        if image.size[0] != reso[0] or image.size[1] != reso[1] or image.mode != "RGB":
            invalid_images += 1
            print(f"{filepath} invalid. (resultion different or color mode non-RGB) [{amount_of_images - count} images remaining]")
            continue
        
        image_list = []
        
        for x in range(image.size[0]):
            row = []
                
            for y in range(image.size[1]):
                row.append(image.getpixel((x,y))[0])

            image_list.append(row)

        image_list.append([directory[1], filename])
        
        images_list.append(image_list)

        print(f"Image set {folder_count} ({folder_count}/{len(images_to_classify_dirs)}) || [{count}/{len(os.listdir(directory[0]))}]            {filepath}")  

    print("\n----------------------------------------------------------------\n")


print(f"Invalidated images: ({invalid_images})\n")

images_list = random.sample(images_list, len(images_list))

print("Images has successfully been shuffled!\n")

cheat_sheet = []

for index, image in enumerate(images_list):
    image_type = image[len(image)-1]
    
    if image_type[0] == "cancer":
        cheat_sheet.append(0)
    elif image_type[0] == "not_cancer":
        cheat_sheet.append(1)
    else:
        print(f"[{index}] image contains invalid ticker.")
        images_list.pop(index)
        continue
    
    image_list_names.append(image_type[1])
    image.pop(len(image)-1)

print("cheatsheet has been created!")

loading_images_time = (perf_counter() - loading_images_time)

# Running Images Through Models

new_folder = f"Comprehensive Test #{randrange(10000)}"

os.mkdir(f"test_results/{new_folder}")

print(f"\noutput folder: {new_folder}\n")

accuracies = []
undetermined_list = []
total_times = []
rights_wrongs = []

total_prediction_time = 0

for count, (model, model_name) in enumerate(zip(models, model_names)):
    print(f"Testing {model_name}...")
    
    p_start_time = perf_counter()
    predictions = model.predict(images_list, verbose=2)       
    prediction_time = round((perf_counter() - p_start_time) * 1000) / 1000

    total_prediction_time = total_prediction_time + prediction_time

    with open(f"test_results/{new_folder}/{model_name}.txt", "a") as f:        
        f.write(f"Brain Cancer Accuracy Test Report\n\nModel: {model_name}\n")      
        f.write(f"\nTotal Prediction Time: {datetime.timedelta(seconds = round(prediction_time))}\nTime-Per-Image: {round(prediction_time / len(images_list), 2)} second(s)\n\nImages: {len(images_list)}\n\n")     

        accuracy, undetermined, right, wrong = getModelAccuracy(predictions, cheat_sheet)

        accuracies.append(accuracy)
        undetermined_list.append(undetermined)
        total_times.append(prediction_time)
        rights_wrongs.append([right, wrong])

        f.write(f"accuracy: {round(accuracy, 2)}%\n") 
        f.write(f"define_rate: {round(((len(images_list) - undetermined) / len(images_list) * 100), 2)}%\n") 
        f.write(f"undetermined_range: {undetermined_range}\n") 

        f.write(f"\nundetermined: {undetermined} images || right: {right} images || wrong: {wrong} images")      

        f.write(f"\n\nRaw Data:\n\n")      

        for prediction, anwser, filename in zip(predictions, cheat_sheet, image_list_names):
            save = f"{filename} || Raw output: {prediction} || Actual Tag: {getAnwserTag(anwser)} || {getAccuracyTag(prediction, anwser)}\n"
            f.write(save)   

    print(f"{model_name} Tested! [{datetime.timedelta(seconds = round(prediction_time))}]         -->         ({count + 1}/{len(models)})\n")

print("\n---------------------------------------------\n")

print("Building Conclusion Folder...\n")

with open(f"test_results/{new_folder}/conclusion.txt", 'a') as f:
    f.write(f"Brain Cancer Accuracy Test Report Conclusion\n")
    f.write(f"\nUndetermined Range: {undetermined_range}")
    f.write(f"\n\nloading_models_time: {datetime.timedelta(seconds = round(loading_models_time))}")
    f.write(f"\nimage_loading_time: {datetime.timedelta(seconds = round(loading_images_time))}")
    f.write(f"\ntotal_prediction_time: {datetime.timedelta(seconds = round(total_prediction_time))}")
    f.write(f"\n\ntotal__time: {datetime.timedelta(seconds = round(total_prediction_time + loading_images_time + loading_models_time))}\n")

    for model_name, accuracy, undetermined, prediction_time, right_wrong in zip(model_names, accuracies, undetermined_list, total_times, rights_wrongs):
        f.write(f"\n\n{model_name} stats:           \n\nprediction_time: {datetime.timedelta(seconds=round(prediction_time))} \ntime_per_image: {round(prediction_time / len(images_list), 2)} second(s) \naccuracy: {round(accuracy, 2)}% \ndefine_rate: {round(((len(images_list) - undetermined) / len(images_list) * 100), 2)}% \nundetermined: {undetermined} \nright: {right_wrong[0]} \nwrong: {right_wrong[1]}\n")

print("Finished Building Conclusion File!")
print(f"\nAll data has been saved in the {new_folder} folder!")

if False:
    os.system("shutdown /s")
