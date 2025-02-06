import os
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random


DATASET_PATH = "ESC-50/audio/"
metadata = "ESC-50/meta/esc50.csv"


def extract_features(file_path, max_pad_len=50):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

 Load dataset and extract features
X, y = [], []
df = pd.read_csv(metadata)
for i, row in df.iterrows():
    file_path = os.path.join(DATASET_PATH, row['filename'])
    feature = extract_features(file_path)
    if feature is not None:
        X.append(feature)
        y.append(row['category'])


X = np.array(X)


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


X = X.reshape(X.shape[0], X.shape[1], X.shape[2])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    tf.keras.Input(shape=(40, 50)), 
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(set(y)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


epochs = 30
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.show()

random_idx = random.randint(0, len(X_test) - 1)  
sample_audio = X_test[random_idx]  
true_label = label_encoder.inverse_transform([y_test[random_idx]])[0]  

sample_audio = sample_audio.reshape(1, sample_audio.shape[0], sample_audio.shape[1])


predicted_probabilities = model.predict(sample_audio)
predicted_label = label_encoder.inverse_transform([np.argmax(predicted_probabilities)])[0]


print(f"True label: {true_label}")
print(f"Predicted label: {predicted_label}")


plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.feature.mfcc(y=librosa.load(os.path.join(DATASET_PATH, df.iloc[random_idx]['filename']))[0], sr=22050, n_mfcc=40), x_axis='time')
plt.colorbar()
plt.title(f'MFCC of {true_label}')
plt.show()


