import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from export_results import export_report
from sklearn.utils import class_weight

# === Config ===
dataset_path = 'dataset/'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
BASE_EPOCHS = 10
EPOCHS = 30
GAUSSIAN_STD = 0.00
SALT_AND_PEPPER_AMT = 0.00
SALT_VS_PEPPER = 0.5

# === Load and Preprocess Images ===
images = []
labels = []
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

for class_name in class_names:
    class_folder = os.path.join(dataset_path, class_name)
    print(f"Loading images from class: {class_name}")
    for filename in os.listdir(class_folder):
        if filename.lower().endswith(image_extensions):
            img_path = os.path.join(class_folder, filename)
            try:
                img = Image.open(img_path).convert('RGB').resize(IMAGE_SIZE)
                images.append(np.array(img))
                labels.append(class_names.index(class_name))
            except Exception as e:
                print(f"Could not load {img_path}: {e}")

# === Convert to NumPy Arrays and Normalize ===
images = np.array(images).astype('float32') / 255.0
labels = np.array(labels)
print(f"Loaded {len(images)} images from {len(class_names)} classes.")

# === Split Dataset ===
X_trainval, X_test, y_trainval, y_test = train_test_split(
    images, labels, test_size=0.1, random_state=42, stratify=labels)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval)

# === Find Class Weights ===
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# === Noise Functions ===
def add_gaussian_noise(images, mean=0.0, std=GAUSSIAN_STD):
    noise = np.random.normal(loc=mean, scale=std, size=images.shape)
    return np.clip(images + noise, 0., 1.)

def add_salt_pepper_noise(images, amount=SALT_AND_PEPPER_AMT, s_vs_p=SALT_VS_PEPPER):
    noisy = images.copy()
    for i in range(len(noisy)):
        img = noisy[i]
        num_pixels = img.size
        num_salt = int(amount * num_pixels * s_vs_p)
        num_pepper = int(amount * num_pixels * (1 - s_vs_p))

        # Salt Noise
        coords = [np.random.randint(0, s, num_salt) for s in img.shape]
        img[tuple(coords)] = 1

        # Pepper Noise
        coords = [np.random.randint(0, s, num_pepper) for s in img.shape]
        img[tuple(coords)] = 0

    return noisy

# === Apply Noise to Training Data ===
X_train = add_gaussian_noise(X_train, std=GAUSSIAN_STD)
X_train = add_salt_pepper_noise(X_train, amount=SALT_AND_PEPPER_AMT)

# === Convert to TensorFlow Datasets ===
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# === Define TensorFlow Data Augmentation ===
data_augmentation = tf.keras.Sequential([
    # tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.05),
    tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.05)),
])

# === Apply Augmentation to Training Set ===
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# === Define Base Model ===
base_model = keras.applications.EfficientNetB0(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False 

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(class_names), activation='softmax')
])

# === Compile Base Model ===
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Train Base Model ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=BASE_EPOCHS,
    class_weight=class_weights,
)

# === Train Deeper Layers ===
base_model.trainable = True

# Fine tune only the top 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with a **lower learning rate**
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Fine-tune the entire model
fine_tune_history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])

# === Evaluate Model ===
test_loss, test_acc= model.evaluate(test_ds)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Combine accuracy and val_accuracy
acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']

# Plot full training history
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.axvline(x=BASE_EPOCHS - 1, color='gray', linestyle='--', label='Fine-tuning Start')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# === Save Model ===
# model.save('micropix.keras')

# === Export Results ===
class_weights = {k: round(float(v), 3) for k, v in class_weights.items()}

augmentation_summary = (
    "RandomFlip(none), "
    "RandomRotation(0.05 radians), "
    "RandomZoom(0.05), "
    "RandomContrast(0.05), "
    "RandomBrightness(0.05)"
)

config = {
    'Image Size': IMAGE_SIZE,
    'Batch Size': BATCH_SIZE,
    'Number of Classes': len(class_names),
    'Class Weights': class_weights,
    'Base Epochs': BASE_EPOCHS,
    'Epochs': EPOCHS,
    'Data Augmentation': augmentation_summary,
    'Gaussian Noise STD': GAUSSIAN_STD,
    'Salt-Pepper Noise Amount': SALT_AND_PEPPER_AMT,
    'Train Size': len(X_train),
    'Validation Size': len(X_val),
    'Test Size': len(X_test),
    'Comments': 'Using EfficientNetB0, fine tune only the top 20 layers, actually removed gaussian and salt & pepper',
}

export_report(config, model, history, fine_tune_history, BASE_EPOCHS, test_acc, filename='micropix_report.pdf')

