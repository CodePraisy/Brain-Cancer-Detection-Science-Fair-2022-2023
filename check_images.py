import os

from PIL import Image

training = "dataset/Training"
testing = "dataset/Testing"
training_dir = os.listdir(training)
testing_dir = os.listdir(testing)

min_res = [9999, 9999]

for dir in training_dir:
    
    print(dir)

    try:
        path = os.path.join(training,dir)
        image_dir = os.listdir(path)
                
        for img in image_dir:
            print(img)
            
            image = Image.open(img)
            if image.size[0] < min_res[0]: min_res[0] = image.size[0]
            if image.size[1] < min_res[1]:	min_res[1] = image.size[1]
    
            print(image.size)
    except Exception as error:
        print(error)     
        
print(str(min_res))