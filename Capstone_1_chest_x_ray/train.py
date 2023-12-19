import os
import time

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.layers import RandomFlip, RandomTranslation, RandomZoom
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Defining parameters for fine-tuning
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
early_stopping = EarlyStopping('val_loss', patience=15, restore_best_weights=True)

IMG_SIZE = 224
BATCH_SIZE = 16

base_dir = './Capstone_1_chest_x_ray/Dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

train_datagen = ImageDataGenerator(rescale = 1.0/255.)
val_datagen = ImageDataGenerator(rescale = 1.0/255.)
test_datagen = ImageDataGenerator(rescale = 1.0/255.)

train_data = train_datagen.flow_from_directory(train_dir, batch_size=BATCH_SIZE, interpolation="bilinear",
                                               class_mode='categorical', classes=['normal', 'pneumonia'], 
                                               target_size=(IMG_SIZE, IMG_SIZE))
val_data = val_datagen.flow_from_directory(validation_dir, batch_size=BATCH_SIZE, interpolation="bilinear",
                                           class_mode  = 'categorical', classes=['normal', 'pneumonia'], 
                                           target_size = (IMG_SIZE, IMG_SIZE))
test_data = test_datagen.flow_from_directory(test_dir, batch_size=BATCH_SIZE, interpolation="bilinear",
                                             class_mode  = 'categorical', classes=['normal', 'pneumonia'], 
                                             target_size = (IMG_SIZE, IMG_SIZE))

base_model = DenseNet201(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the pretrained weights
base_model.trainable = False

# Augmantation sequence
intput_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
augmantation = Sequential([
    RandomFlip("horizontal"),
    RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
    RandomZoom(0.2)
])
x = augmantation(intput_layer)
# pretrain output
pretrain_out = base_model(x, training = False)

# Rebuild top
x = GlobalAveragePooling2D()(pretrain_out)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)

# Add a dropout rate 
x = Dropout(0.4)(x)

# Output layer
outputs = Dense(2, activation='softmax')(x)

# Compile
model_densenet = tf.keras.Model(inputs=intput_layer, outputs=outputs, name="DenseNet201")
model_densenet.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

def unfreeze_model(model, optimizer):
    # We unfreeze all layers, leaving the BatchNorm layers frozen to preserve model knowledge.
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

optimizer_densenet_ft = Adam(learning_rate=1e-4)
unfreeze_model(model_densenet, optimizer_densenet_ft)

history_densenet_ft = model_densenet.fit(train_data, validation_data = val_data, epochs=100, callbacks=[reduce_lr, early_stopping])