import pandas as pd
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Function to extract MFCC features from audio files
def extract_mfcc_features(filename):
    data, sample_rate = librosa.load(filename)
    mfccs_features = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Function to check label for audio file
def check_label(filepath):
    filename = os.path.basename(filepath)
    file_num = int(filename.split('.')[0])
    file_label = df[df['itemid'] == file_num]['hasbird'].values[0]
    if file_label == 1:
        return 'Has Bird'
    else:
        return 'No Bird'

# Load metadata
df = pd.read_csv('ff1010bird_metadata.csv')

# Extracting features and labels
features = []
labels = []
for dirname, _, filenames in os.walk('ff1010bird_wav/wav'):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        mfccs_scaled_features = extract_mfcc_features(file_path)
        features.append(mfccs_scaled_features)
        labels.append(check_label(file_path))

# Encoding labels
labelencoder = LabelEncoder()
labels_encoded = labelencoder.fit_transform(labels)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Training Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Evaluation
y_pred_decision_tree = decision_tree.predict(X_test)
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
print("Decision Tree Classifier Accuracy:", accuracy_decision_tree)

# Save Decision Tree model
model_filename = 'decision_tree_model.sav'
import joblib
joblib.dump(decision_tree, model_filename)

# Save Label Encoder
label_encoder_filename = 'dec_label_encoder.pkl'
import pickle
with open(label_encoder_filename, 'wb') as le_file:
    pickle.dump(labelencoder, le_file)

# Plot Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, filled=True, feature_names=[f'MFCC_{i}' for i in range(1, 41)])
plt.savefig('decision_tree_plot.png')
plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate Precision
precision = precision_score(y_test, y_pred_decision_tree)

# Calculate Recall
recall = recall_score(y_test, y_pred_decision_tree)

# Calculate F1 Score
f1 = f1_score(y_test, y_pred_decision_tree)

print("Decision Tree Classifier Metrics:")
print(f"Accuracy: {accuracy_decision_tree}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate Confusion Matrix
cm_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_decision_tree, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.xticks(ticks=[0.5, 1.5], labels=['No Bird', 'Has Bird'])
plt.yticks(ticks=[0.5, 1.5], labels=['No Bird', 'Has Bird'])
plt.show()
