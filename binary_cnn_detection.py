import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


# binary dataset
binary_dataset_path = 'dataset_binary'

if not os.path.exists(binary_dataset_path):
    os.makedirs(binary_dataset_path)
    os.makedirs(os.path.join(binary_dataset_path, 'Training'))
    os.makedirs(os.path.join(binary_dataset_path, 'Testing'))
    os.makedirs(os.path.join(binary_dataset_path, 'Training', 'tumor'))
    os.makedirs(os.path.join(binary_dataset_path, 'Training', 'no_tumor'))
    os.makedirs(os.path.join(binary_dataset_path, 'Testing', 'tumor'))
    os.makedirs(os.path.join(binary_dataset_path, 'Testing', 'no_tumor'))

    for folder in ['Training', 'Testing']:
        for label in ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']:
            src_folder = os.path.join('dataset', folder, label)
            dst_folder = os.path.join(binary_dataset_path, folder, 'tumor')
            for img_file in os.listdir(src_folder):
                shutil.copy(os.path.join(src_folder, img_file), os.path.join(dst_folder, img_file))
        
        src_folder = os.path.join('dataset', folder, 'no_tumor')
        dst_folder = os.path.join(binary_dataset_path, folder, 'no_tumor')
        for img_file in os.listdir(src_folder):
            shutil.copy(os.path.join(src_folder, img_file), os.path.join(dst_folder, img_file))

binary_train_path = os.path.join(binary_dataset_path, 'Training')
binary_test_path = os.path.join(binary_dataset_path, 'Testing')

# model archeticture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(75, 75, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# compile
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training data generator
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    binary_train_path,
    target_size=(75, 75),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    binary_train_path,
    target_size=(75, 75),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation')

#train
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=30)
