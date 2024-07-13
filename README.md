# ChinX-Gender: Gender Prediction from Chin X-Ray Images

## Overview

ChinX-Gender is a deep learning project that aims to predict gender (male or female) from chin x-ray images. The project leverages convolutional neural networks (CNNs) to classify the images.

## Dataset

The dataset used in this project consists of chin x-ray images divided into two categories:

- `E`: Male
- `K`: Female

The images are stored in the following directory structure:

```
/gdrive/MyDrive/Dersler/Lisans/Veri_Madenciligi/CTanima
  ├── E
  └── K
```

## Project Steps

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/gdrive/')
```

### 2. Import Libraries and Setup

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
```

### 3. Load and Preprocess Data

```python
DATADIR = "/gdrive/MyDrive/Dersler/Lisans/Veri_Madenciligi/CTanima"
CATEGORIES = ["E","K"]
IMG_SIZE = 100

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)
```

### 4. Save Processed Data

```python
with open("X.pickle", "wb") as pickle_out:
    pickle.dump(X, pickle_out)

with open("y.pickle", "wb") as pickle_out:
    pickle.dump(y, pickle_out)
```

### 5. Load Processed Data

```python
with open("X.pickle", "rb") as pickle_in:
    X = pickle.load(pickle_in)

with open("y.pickle", "rb") as pickle_in:
    y = pickle.load(pickle_in)
```

### 6. Normalize Data

```python
X = X / 255.0
```

### 7. Build the Model

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.summary()
```

### 8. Compile the Model

```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

### 9. Train the Model

```python
model.fit(X, y, batch_size=16, epochs=15, verbose=1, validation_split=0.1)
```

### 10. Save the Model

```python
model.save('/test_model.h5')
```

### 11. Test the Model

```python
DATADIRtest = "/gdrive/MyDrive/Dersler/Lisans/Veri_Madenciligi/CTanima/test"
CATEGORIEStest = ["E","K"]

test_data = []

def create_test_data():
    for category in CATEGORIEStest:
        path = os.path.join(DATADIRtest, category)
        class_num = CATEGORIEStest.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([new_array, class_num])
            except Exception as e:
                pass

create_test_data()

Xtest = []
ytest = []

for features, label in test_data:
    Xtest.append(features)
    ytest.append(label)

Xtest = np.array(Xtest).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Xtest = Xtest / 255.0

score = model.evaluate(Xtest, ytest, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])
```

## Requirements

- Google Colab
- TensorFlow
- Keras
- OpenCV
- Numpy
- Matplotlib
- Pickle

## Conclusion

ChinX-Gender successfully classifies chin x-ray images into male and female categories using a CNN model. The model is trained and tested with a decent accuracy, demonstrating its potential for practical applications.
