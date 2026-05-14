# # import tensorflow as tf
# # import os

# # # Specify the path to the .h5 model file
# # model_path = '/home/cr7karki/Downloads/final_model.h5'

# # # Print the absolute path to verify it's correct
# # print(f"Loading model from: {os.path.abspath(model_path)}")

# # # Load the Keras model (.h5 format)
# # try:
# #     model = tf.keras.models.load_model(model_path)
# #     print(f"Model loaded successfully from {model_path}")

# #     # Specify the directory where the SavedModel should be saved
# #     saved_model_path = '/home/cr7karki/Downloads/onlyres_saved_model'

# #     # Save the model in TensorFlow's SavedModel format
# #     model.save(saved_model_path, save_format='tf')
# #     print(f"Model successfully converted and saved to: {saved_model_path}")

# # except Exception as e:
# #     print(f"Error loading or saving the model: {e}")

# import tensorflow as tf
# import os
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Concatenate, UpSampling2D, Conv2D, Reshape


# # Define your custom model
# def ResNet_model():
#     model = ResNet50(weights=None)  # Initialize ResNet50 without pre-trained weights

#     # Set layers to be trainable as needed
#     for layer in model.layers[:-10]:
#         layer.trainable = True

#     # Define the additional layers you need (concatenation, upsampling, etc.)
#     block1 = model.get_layer("conv1_relu").output
#     block2 = model.get_layer("conv2_block3_out").output
#     block3 = model.get_layer("conv3_block4_out").output
#     block4 = model.get_layer("conv4_block6_out").output
#     block5 = model.get_layer("conv5_block3_out").output

#     x = Concatenate()([UpSampling2D()(block5), block4])
#     x = Conv2D(100, (1, 1), activation='relu')(x)
#     x = Concatenate()([UpSampling2D()(x), block3])
#     x = Conv2D(100, (1, 1), activation='relu')(x)
#     x = Concatenate()([UpSampling2D()(x), block2])
#     x = Conv2D(100, (1, 1), activation='relu')(x)
#     x = Concatenate()([UpSampling2D()(x), block1])
#     x = Conv2D(100, (1, 1), activation='relu')(x)
#     x = UpSampling2D()(x)
#     x = Conv2D(1, kernel_size=1, strides=1, activation="sigmoid")(x)
#     x = Reshape((224, 224, 1))(x)

#     return Model(inputs=model.input, outputs=x)

# # Specify the path to the .h5 weights file
# weights_path = '/home/cr7karki/Downloads/final_model.h5'

# # Specify the directory where the SavedModel should be saved
# saved_model_path = '/home/cr7karki/Downloads/saved_model'

# # Print the absolute path to verify it's correct
# print(f"Loading model from: {os.path.abspath(weights_path)}")

# try:
#     # Initialize the model architecture
#     model = ResNet_model()

#     # Load the weights into the model
#     model.load_weights(weights_path)
#     print(f"Model weights loaded successfully from {weights_path}")

#     # Save the model in TensorFlow's SavedModel format
#     model.save(saved_model_path)
#     print(f"Model successfully saved in SavedModel format at: {saved_model_path}")

# except Exception as e:
#     print(f"Error loading or saving the model: {e}")


import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, UpSampling2D, Conv2D, Reshape

# Define your custom model
def ResNet_model():
    model = ResNet50(weights=None)  # Initialize ResNet50 without pre-trained weights

    # Set layers to be trainable as needed
    for layer in model.layers[:-10]:
        layer.trainable = True

    # Define the additional layers you need (concatenation, upsampling, etc.)
    block1 = model.get_layer("conv1_relu").output
    block2 = model.get_layer("conv2_block3_out").output
    block3 = model.get_layer("conv3_block4_out").output
    block4 = model.get_layer("conv4_block6_out").output
    block5 = model.get_layer("conv5_block3_out").output

    x = Concatenate()([UpSampling2D()(block5), block4])
    x = Conv2D(100, (1, 1), activation='relu')(x)
    x = Concatenate()([UpSampling2D()(x), block3])
    x = Conv2D(100, (1, 1), activation='relu')(x)
    x = Concatenate()([UpSampling2D()(x), block2])
    x = Conv2D(100, (1, 1), activation='relu')(x)
    x = Concatenate()([UpSampling2D()(x), block1])
    x = Conv2D(100, (1, 1), activation='relu')(x)
    x = UpSampling2D()(x)
    x = Conv2D(1, kernel_size=1, strides=1, activation="sigmoid")(x)
    x = Reshape((224, 224, 1))(x)

    return Model(inputs=model.input, outputs=x)

# Specify the path to the .h5 weights file
weights_path = '/home/cr7karki/Downloads/final_res_model.h5'

# Specify the directory where the SavedModel should be saved
saved_model_path = '/home/cr7karki/Downloads/saved_res_model_dir'

# Print the absolute path to verify it's correct
print(f"Loading model from: {os.path.abspath(weights_path)}")

try:
    # Initialize the model architecture
    model = ResNet_model()

    # Load the weights into the model
    model.load_weights(weights_path)
    print(f"Model weights loaded successfully from {weights_path}")

    # Save the model in TensorFlow's SavedModel format
    tf.saved_model.save(model, saved_model_path)  # Explicitly save in SavedModel format
    print(f"Model successfully saved in SavedModel format at: {saved_model_path}")

except Exception as e:
    print(f"Error loading or saving the model: {e}")
