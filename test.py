import tensorflow as tf

# Load the SavedModel
model = tf.saved_model.load("/home/cr7karki/Documents/projects/samir/conversion/saved_res_model_dir")


# Check the input and output signatures
print("Model input signature:", model.signatures['serving_default'].inputs)
print("Model output signature:", model.signatures['serving_default'].outputs)


