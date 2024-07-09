import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa
import librosa.display
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os


# data processing, CSV file I/O (e.g. pd.read_csv)
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('ff1010bird_metadata.csv')
print(f'Number of Labels: {df.shape[0]}')
df.head()

print('Number of audios with bird sounds: {}'.format(df[df['hasbird'] == 1].shape[0]))
print('Number of audios without bird sounds: {}'.format(df[df['hasbird'] == 0].shape[0]))

filepath = "ff1010bird_wav/wav/64486.wav"
ipd.Audio(filepath)

#Check for  corresponding audio label

def check_label(filepath):
    # Extract the filename from the filepath
    filename = os.path.basename(filepath)
    # Split the filename to get the file number
    file_num = int(filename.split('.')[0])
    file_label = df[df['itemid'] == file_num].hasbird.values[0]
    if file_label == 1:
        file_label = 'Has Bird'
    else:
        file_label = 'No Bird'
    return file_label

#Audio waveplot

data, sample_rate = librosa.load(filepath)
file_label = check_label(filepath)
plt.figure(figsize=(12, 5))
plt.title(f'Waveplot: {file_label}')

#Audio With Bird
filepath = "ff1010bird_wav/wav/100.wav"

#Waveplot
file_label = check_label(filepath)
data, sample_rate = librosa.load(filepath)
plt.figure(figsize=(12, 5))
plt.title(f'Waveplot: {file_label}')

#Play Audio
print('Audio')
ipd.Audio(filepath)

data1, sample_rate1 = librosa.load('ff1010bird_wav/wav/100.wav')
plt.figure(figsize=(20, 10))
D = librosa.amplitude_to_db(np.abs(librosa.stft(data1)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram - Bird')


data1, sample_rate1 = librosa.load("ff1010bird_wav/wav/64486.wav")
plt.figure(figsize=(20, 10))
D = librosa.amplitude_to_db(np.abs(librosa.stft(data1)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram - No Bird')

mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
print(mfccs.shape)
print(mfccs)

np.mean(mfccs.T,axis=0)

np.mean(mfccs.T,axis=0).shape

#We define a function to extract the mfcc features from the audio
def extract_mfcc_features(filename):
    #Load audio file
    data1, sample_rate1 = librosa.load(filename)
    #Extract mfcc features
    mfccs_features = librosa.feature.mfcc(y=data1, sr=sample_rate1, n_mfcc=40)
    #in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features




#We now extract the mfcc features from the audios we have
features = []
labels = []
for dirname, _, filenames in os.walk('ff1010bird_wav/wav'):
    for filename in filenames:
        my_filename = os.path.join(dirname, filename)
        mfccs_scaled_features = extract_mfcc_features(my_filename)
        features.append(mfccs_scaled_features)
        file_label = check_label(my_filename)
        labels.append(file_label)



print(f'Num Features: {len(features)}')
print(f'Num Labels: {len(labels)}')
#Create a DataFrame consisting of features and labels
df = pd.DataFrame({'MFCCs':features,
                  'labels':labels })


df['labels'].value_counts()


#Visualizing the distribution of data for the 2 classes
sns.countplot(df, x="labels")
plt.title('Samples per Class')

# - There is class imbalance as the number of samples in the 'No Bird' class is greater than those in the 'Has Bird' class. We therefore need to downsample the 'No Bird' class

majority_class = df['labels'].value_counts().idxmax()
minority_class = df['labels'].value_counts().idxmin()
print(f'Majority Class: {majority_class}')
print(f'Minority Class: {minority_class}')

# Number of samples to remove from the majority class
minority_class_count = len(df[df['labels'] == minority_class])
majority_class_count = len(df[df['labels'] == majority_class])


# Randomly remove samples from the majority class
df_downsampled = df[df['labels'] == majority_class].sample(n=majority_class_count, replace=False)

# Combine the downsampled majority class with the minority class
df_balanced = pd.concat([df_downsampled, df[df['labels'] == minority_class]])

df_balanced['labels'].value_counts()

#Visualizing the distribution of data for the 2 classes
sns.countplot(df_balanced, x="labels")
plt.title('Samples per Class')

# - Class Balance has now been achieved

x = np.array(df_balanced['MFCCs'].tolist())
y = df_balanced['labels']


print(x.shape)
x[:1]

print(y.shape)
y.head()


labelencoder=LabelEncoder()


y = to_categorical(labelencoder.fit_transform(y))
y[:5] #View the first 5 samples


y.shape


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print(X_train.shape)
print(X_test.shape)


X_train = X_train.reshape(6152, 10, 4, 1)
X_test = X_test.reshape(1538, 10, 4, 1)


input_dim = (10, 4, 1)


# Necessary libraries


model = Sequential()
model.add(Conv2D(64, (3, 3), padding = "same", activation = "relu", input_shape = input_dim))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation = "relu"))
model.add(Dense(2, activation = "softmax"))


#Visualizing the Model Architecture
import pydot
tf.keras.utils.plot_model(model)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# We Create a ModelCheckpoint callback to save the best model based on validation accuracy

checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_acc', save_best_only=True)

#history = model.fit(X_train, y_train, epochs = 30, batch_size = 50, validation_data = (X_test, y_test))
history = model.fit(X_train, y_train, epochs = 50, batch_size = 50, validation_data = (X_test, y_test))


#Plotting the loss curves
pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.title('Training Curves')
plt.show()

predictions = model.predict(X_test)
score = model.evaluate(X_test, y_test)
print(score)


preds = np.argmax(predictions, axis = 1)
y1 = np.argmax(y_test, axis = 1)
preds



def classify_audio(filename):
    #preprocess the audio file
    audio, sample_rate = librosa.load(filename)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    #Reshape MFCC feature to 2-D array
    mfccs_scaled_features=mfccs_scaled_features.reshape(1, 10, 4, 1)
    #predicted_label=model.predict_classes(mfccs_scaled_features)
    x_predict=model.predict(mfccs_scaled_features)
    predicted_label=np.argmax(x_predict,axis=1)
    #print(predicted_label)
    prediction_class = labelencoder.inverse_transform(predicted_label) [0]
    if prediction_class == 'Oxpecker':
        print(f'Prediction Probability: {x_predict[0][0]}')
    else:
        print(f'Prediction Probability: {x_predict[0][1]}')
    print(f'Predicted Class: {prediction_class}')



classify_audio('ff1010bird_wav/wav/100275.wav')
model.save("Bird_Detector.h5")


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define input dimension for CNN
input_dim = (10, 4, 1)

# Define the CNN model
model = Sequential()
model.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_dim))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dense(2, activation="softmax"))

# Compile the CNN model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
history = model.fit(X_train, y_train, epochs=50, batch_size=50, validation_data=(X_test, y_test))

# Evaluate the CNN model
_, cnn_accuracy = model.evaluate(X_test, y_test)
print("CNN Model Accuracy:", cnn_accuracy)

# Extract features from the CNN model
cnn_features_train = model.predict(X_train)
cnn_features_test = model.predict(X_test)

# Define and train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(cnn_features_train, np.argmax(y_train, axis=1))

# Predict using the SVM model
svm_predictions = svm_model.predict(cnn_features_test)

# Calculate accuracy of the SVM model
svm_accuracy = accuracy_score(np.argmax(y_test, axis=1), svm_predictions)
print("CNN+SVM Model Accuracy:", svm_accuracy)
# Save the combined CNN-SVM model as cnnsvm.h5
model.save("cnnsvm.h5")

#Saving to h5


#Converting to tflite
model = tf.keras.models.load_model('Bird_Detector.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


with open('Bird_Detector.tflite', 'wb') as f:
    f.write(tflite_model)


# Initialize the label encoder
labelencoder = LabelEncoder()

# Fit the label encoder to the target labels
labelencoder.fit(labels)

# Save the label encoder classes to a numpy file
np.save('labelencoder_classes.npy', labelencoder.classes_)



from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# CNN Model Metrics
cnn_accuracy = accuracy_score(np.argmax(y_test, axis=1), preds)
cnn_f1 = f1_score(np.argmax(y_test, axis=1), preds)
cnn_precision = precision_score(np.argmax(y_test, axis=1), preds)
cnn_recall = recall_score(np.argmax(y_test, axis=1), preds)

print("CNN Model Metrics:")
print(f"Accuracy: {cnn_accuracy}")
print(f"F1 Score: {cnn_f1}")
print(f"Precision: {cnn_precision}")
print(f"Recall: {cnn_recall}")
print()


# Make predictions using the combined CNN+SVM model
cnnsvm_predictions = model.predict(X_test)  # Assuming 'model' refers to your combined CNN+SVM model

# CNN+SVM Model Metrics
cnnsvm_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(cnnsvm_predictions, axis=1))
cnnsvm_f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(cnnsvm_predictions, axis=1))
cnnsvm_precision = precision_score(np.argmax(y_test, axis=1), np.argmax(cnnsvm_predictions, axis=1))
cnnsvm_recall = recall_score(np.argmax(y_test, axis=1), np.argmax(cnnsvm_predictions, axis=1))

print("CNN+SVM Model Metrics:")
print(f"Accuracy: {cnnsvm_accuracy}")
print(f"F1 Score: {cnnsvm_f1}")
print(f"Precision: {cnnsvm_precision}")
print(f"Recall: {cnnsvm_recall}")

from sklearn.metrics import confusion_matrix

# Calculate confusion matrix for CNN+SVM model
svm_cm = confusion_matrix(np.argmax(y_test, axis=1), svm_predictions)
print("Confusion Matrix (CNN+SVM Model):")
print(svm_cm)

# Calculate confusion matrix for CNN model
cnn_cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
print("Confusion Matrix (CNN Model):")
print(cnn_cm)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define labels
labels = ['Bird', 'No Bird']

# Confusion Matrix for CNN+SVM Model
svm_cm = np.array([[167, 218],
                   [130, 1023]])

# Confusion Matrix for CNN Model
cnn_cm = np.array([[125, 260],
                   [ 63, 1090]])

# Plot Confusion Matrices
plt.figure(figsize=(12, 6))

# Plot Confusion Matrix for CNN+SVM Model
plt.subplot(1, 2, 1)
sns.heatmap(svm_cm, annot=True, cmap="Blues", fmt="d", cbar=False, xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (CNN+SVM Model)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Plot Confusion Matrix for CNN Model
plt.subplot(1, 2, 2)
sns.heatmap(cnn_cm, annot=True, cmap="Blues", fmt="d", cbar=False, xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (CNN Model)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()