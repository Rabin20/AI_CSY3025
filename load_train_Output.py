from keras.models import load_model

# Load the saved model
loaded_model = load_model('facedetection.h5')

# Access the loaded model object
print(loaded_model.summary())

