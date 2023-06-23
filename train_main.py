import os
import cv2
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Dinh nghia cac bien
gestures = {'L_': 'L',
           'fi': 'Nam tay',
           'ok': 'OK',
           'pe': 'HI',
           'pa': 'Hello'
            }

gestures_map = {'Nam tay': 0,
                'L': 1,
                'OK': 2,
                'HI': 3,
                'Hello': 4
                }

gesture_names = {0: 'Nam tay',
                 1: 'L',
                 2: 'OK',
                 3: 'HI',
                 4: 'Hello'
                 }

image_path = 'data'
models_path = 'models/saved_model.hdf5'
imageSize = 224

# Ham xu ly anh resize ve 224x224 va chuyen ve numpy array
def process_image(path):
    img = Image.open(path)
    img = img.resize(imageSize, imageSize)
    img = np.array(img)
    return img

# Xu ly du lieu dau vao
def process_data(X_data, y_data):
    X_data = np.array(X_data, dtype = 'float32')
    X_data = np.stack((X_data,)*3, axis=-1)
    X_data /= 255
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)
    return X_data, y_data

# Ham duyet thu muc anh dung de train
def walk_file_tree(image_path):
    X_data = []
    y_data = []
    for directory, subdirectories, files in os.walk(image_path):
        for file in files:
            if not file.startswith('.'):
                path = os.path.join(directory, file)
                gesture_name = gestures[file[0:2]]
                y_data.append(gestures_map[gesture_name])
                X_data.append(process_image(path))
            else:
                continue
    X_data, y_data = process_data(X_data, y_data)
    return X_data, y_data

# Load du lieu vao X va Y
X_data, y_data = walk_file_tree(image_path)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=12, stratify=y_data)

# Khoi tao model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(imageSize, imageSize, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=224, validation_data=(X_test, y_test), verbose=1)

model.save('models/mymodel1.h5')
