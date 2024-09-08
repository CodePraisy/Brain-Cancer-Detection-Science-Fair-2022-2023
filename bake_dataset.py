import os
import pickle
import random

from PIL import Image

non_cancerous_images = 'cleaned_dataset/no_cancer'
cancerous_images  = 'cleaned_dataset/cancer'

savefile_1 = "list_of_images.pkl"
savefile_2 = "list_of_answers.pkl"

images = []

reso = (256, 256)

invalid_images = 0

for count, filename in enumerate(os.listdir(cancerous_images)):
    try:
        file_path = os.path.join(cancerous_images, filename)
        image = Image.open(file_path)
    except:
        invalid_images += 1
        continue    
    
    if not image.size[0] == reso[0]:
        invalid_images += 1
        continue      
        
    if not image.size[1] == reso[1]:
        invalid_images += 1
        continue

    image_list = []

    for x in range(image.size[0]):
        row = []
            
        for y in range(image.size[1]):
            row.append(image.getpixel((x,y))[0])

        image_list.append(row)

    image_list.append('cancer')
    images.append(image_list)
        
    if round(count / 100) == count / 100:
        print(f"Images || {cancerous_images}: {count}/{len(os.listdir(cancerous_images))}")  

print("\n----------------------------------------------------------------\n")

for count, filename in enumerate(os.listdir(non_cancerous_images)):
    try:
        file_path = os.path.join(non_cancerous_images, filename)
        image = Image.open(file_path)
    except:
        invalid_images += 1
        continue    
    
    if not image.size[0] == reso[0]:
        invalid_images += 1
        continue      
        
    if not image.size[1] == reso[1]:
        invalid_images += 1
        continue

    image_list = []

    for x in range(image.size[0]):
        row = []
            
        for y in range(image.size[1]):
            row.append(image.getpixel((x,y))[0])

        image_list.append(row)

    image_list.append('not_cancer')
    images.append(image_list)


    if round(count / 100) == count / 100:
        print(f"Image set 2 || {non_cancerous_images}: {count}/{len(os.listdir(non_cancerous_images))}")  

print("\n----------------------------------------------------------------\n")

print(f"Invalidated images: ({invalid_images})\n")

images = random.sample(images, len(images))

print("Images has successfully been shuffled!\n")

cheat_sheet = []

for index, image in enumerate(images):
    image_type = image[len(image)-1]
    
    if image_type == "cancer":
        cheat_sheet.append(0)
    elif image_type == "not_cancer":
        cheat_sheet.append(1)
    else:
        print(f"[{index}] image contains invalid ticker.")
        continue

    image.pop(len(image)-1)

print("cheatsheet has been created!")

open_file_1 = open(savefile_1, "wb")
pickle.dump(images, open_file_1)
open_file_1.close()

open_file_2 = open(savefile_2, "wb")
pickle.dump(cheat_sheet, open_file_2)
open_file_2.close()   

print(f"Images has successfully been saved!")