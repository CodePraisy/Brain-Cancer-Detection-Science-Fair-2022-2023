import os
import tensorflow as tf

models_folder = "models"

print("\n------------------------------------------------------------------------\n")

for model_file in os.listdir(models_folder):
    print(f"\n{model_file}\n")
    model = tf.keras.models.load_model(os.path.join(models_folder, model_file))
    print(model.summary())