# ==============================================================================
# Brain Tumor Detection using U-Net
# ==============================================================================
# This script implements a U-Net architecture for brain tumor segmentation
# using the LGG MRI Segmentation Dataset.
#
# ==============================================================================

# ==============================================================================
# 1. Imports and Configuration
# ==============================================================================
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import kagglehub

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration Constants
IM_HEIGHT = 256
IM_WIDTH = 256
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
SMOOTH = 100

# ==============================================================================
# 2. Dataset Loading and Preprocessing
# ==============================================================================
print("Downloading dataset...")
# Download dataset using kagglehub
path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
print("Dataset downloaded to:", path)

# Initialize lists to store paths
image_paths = []
mask_paths = []

# Walk through the directory to find images and masks
for dirpath, dirnames, filenames in os.walk(path):
    for filename in filenames:
        if 'mask' not in filename and filename.endswith('.tif'):
            image_path = os.path.join(dirpath, filename)
            mask_path = os.path.join(dirpath, filename.replace('.tif', '_mask.tif'))
            
            # Check if mask exists
            if os.path.exists(mask_path):
                image_paths.append(image_path)
                mask_paths.append(mask_path)

# Create DataFrame
df = pd.DataFrame({'image_path': image_paths, 'mask_path': mask_paths})
print(f"Total images found: {len(df)}")

# Split into Train+Val and Test
df_train_val, df_test = train_test_split(df, test_size=0.1, random_state=SEED)

# Split Train+Val into Train and Validation
df_train, df_val = train_test_split(df_train_val, test_size=0.2, random_state=SEED)

print(f"Training set: {len(df_train)}")
print(f"Validation set: {len(df_val)}")
print(f"Test set: {len(df_test)}")

# ==============================================================================
# 3. Data Generators
# ==============================================================================
def adjust_data(img, mask):
    """
    Normalize image and mask, and binarize the mask.
    """
    img = img / 255.0
    mask = mask / 255.0
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img, mask)

def data_generator(data_frame, batch_size, aug_dict, image_color_mode="rgb",
                   mask_color_mode="grayscale", image_save_prefix="image",
                   mask_save_prefix="mask", save_to_dir=None, target_size=(256, 256), seed=1):
    """
    Custom data generator using ImageDataGenerator.
    """
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col="image_path",
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col="mask_path",
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img, mask)

# Augmentation parameters for training
train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')

# Create generators
train_gen = data_generator(df_train, BATCH_SIZE, train_generator_args, target_size=(IM_HEIGHT, IM_WIDTH))
val_gen = data_generator(df_val, BATCH_SIZE, dict(), target_size=(IM_HEIGHT, IM_WIDTH))
test_gen = data_generator(df_test, BATCH_SIZE, dict(), target_size=(IM_HEIGHT, IM_WIDTH))

# ==============================================================================
# 4. Metrics and Loss Functions
# ==============================================================================
def dice_coefficients(y_true, y_pred, smooth=SMOOTH):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coefficients_loss(y_true, y_pred):
    return -dice_coefficients(y_true, y_pred)

def iou(y_true, y_pred, smooth=SMOOTH):
    intersection = tf.keras.backend.sum(y_true * y_pred)
    sum_ = tf.keras.backend.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

# ==============================================================================
# 5. U-Net Model Architecture
# ==============================================================================
def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    bn1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(bn1)
    bn1 = BatchNormalization(axis=3)(conv1)
    bn1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    bn2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(bn2)
    bn2 = BatchNormalization(axis=3)(conv2)
    bn2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    bn3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding='same')(bn3)
    bn3 = BatchNormalization(axis=3)(conv3)
    bn3 = Activation('relu')(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    bn4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), padding='same')(bn4)
    bn4 = BatchNormalization(axis=3)(conv4)
    bn4 = Activation('relu')(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    # Bridge
    conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
    bn5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, (3, 3), padding='same')(bn5)
    bn5 = BatchNormalization(axis=3)(conv5)
    bn5 = Activation('relu')(bn5)

    # Decoder
    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), padding='same')(up6)
    bn6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, (3, 3), padding='same')(bn6)
    bn6 = BatchNormalization(axis=3)(conv6)
    bn6 = Activation('relu')(bn6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), padding='same')(up7)
    bn7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, (3, 3), padding='same')(bn7)
    bn7 = BatchNormalization(axis=3)(conv7)
    bn7 = Activation('relu')(bn7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), padding='same')(up8)
    bn8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, (3, 3), padding='same')(bn8)
    bn8 = BatchNormalization(axis=3)(conv8)
    bn8 = Activation('relu')(bn8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), padding='same')(up9)
    bn9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, (3, 3), padding='same')(bn9)
    bn9 = BatchNormalization(axis=3)(conv9)
    bn9 = Activation('relu')(bn9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn9)

    return Model(inputs=[inputs], outputs=[conv10])

# Initialize model
print("Initializing U-Net model...")
model = unet()
model.summary()

# ==============================================================================
# 6. Training
# ==============================================================================
print("Starting training...")
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
              loss=dice_coefficients_loss, 
              metrics=["binary_accuracy", iou, dice_coefficients])

callbacks = [
    ModelCheckpoint('unet_brain_tumor.keras', verbose=1, save_best_only=True)
]

history = model.fit(
    train_gen,
    steps_per_epoch=len(df_train) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=val_gen,
    validation_steps=len(df_val) // BATCH_SIZE
)

# ==============================================================================
# 7. Evaluation and Visualization
# ==============================================================================
print("Training complete. Evaluating...")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['dice_coefficients'], label='Train Dice')
plt.plot(history.history['val_dice_coefficients'], label='Val Dice')
plt.title('Dice Coefficient')
plt.legend()

plt.show()

# Load best model and evaluate on test set
print("Loading best model for evaluation...")
model = load_model('unet_brain_tumor.keras', 
                   custom_objects={'dice_coefficients_loss': dice_coefficients_loss, 
                                   'iou': iou, 
                                   'dice_coefficients': dice_coefficients})

results = model.evaluate(test_gen, steps=len(df_test) // BATCH_SIZE)
print("Test Loss:", results[0])
print("Test IoU:", results[2])
print("Test Dice:", results[3])

# Visualize Predictions
print("Visualizing predictions...")
for i in range(5):
    index = np.random.randint(0, len(df_test))
    row = df_test.iloc[index]
    
    img = cv2.imread(row['image_path'])
    img = cv2.resize(img, (IM_HEIGHT, IM_WIDTH))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IM_HEIGHT, IM_WIDTH))
    mask = mask / 255.0
    
    pred = model.predict(img)[0]
    pred = np.squeeze(pred)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(img))
    plt.title('Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('True Mask')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap='gray')
    plt.title('Predicted Mask')
    plt.show()
