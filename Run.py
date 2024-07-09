import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model('cnnsvm.h5')

# Load the label encoder for decoding predictions
labelencoder = LabelEncoder()
labelencoder.classes_ = np.load('labelencoder_classes.npy')  # Ensure you save the label encoder classes during training

def classify_audio(filename):
    # Preprocess the audio file
    audio, sample_rate = librosa.load(filename)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    # Reshape MFCC feature to 2-D array
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, 10, 4, 1)
    # Predict
    predictions = model.predict(mfccs_scaled_features)
    predicted_label = np.argmax(predictions, axis=1)
    prediction_class = labelencoder.inverse_transform(predicted_label)[0]
    # Display prediction result
    print(f'Predicted Class: {prediction_class}')

# Example usage
classify_audio('ff1010bird_wav/wav/65355.wav')
