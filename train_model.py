import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from export_results import export_report
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === Configuration ===
dataset_path = 'dataset/'
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = IMAGE_SIZE + (3,)
BATCH_SIZE = 32
BASE_EPOCHS = 20
EPOCHS = 30
DROPOUT = 0.6
GAUSSIAN_STD = 0.01
BRIGHTNESS_DELTA = 0.2
FLIP_MODE = "horizontal"
ROTATION_FACTOR = 0.2
ZOOM_FACTOR = 0.2
CONTRAST_FACTOR = 0.2

# === Data Loading and Initial Preprocessing ===
images = []
labels = []
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
num_classes = len(class_names)

for class_idx, class_name in enumerate(class_names):
    class_folder = os.path.join(dataset_path, class_name)
    print(f"Loading images from class: {class_name}")
    for filename in os.listdir(class_folder):
        if filename.lower().endswith(image_extensions):
            img_path = os.path.join(class_folder, filename)
            try:
                img = Image.open(img_path).convert('RGB').resize(IMAGE_SIZE)
                images.append(np.array(img))
                labels.append(class_idx)
            except Exception as e:
                print(f"Could not load {img_path}: {e}")

# === Convert to NumPy Arrays ===
images = np.array(images).astype('float32')
labels = np.array(labels)
print(f"Loaded {len(images)} images from {num_classes} classes.")

# === Dataset Splitting ===
X_trainval, X_test, y_trainval, y_test = train_test_split(
    images, labels, test_size=0.1, random_state=42, stratify=labels)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval)

# === Class Weight Calculation ===
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# === Data Augmentation Pipeline Definition ===
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(FLIP_MODE),
    tf.keras.layers.RandomRotation(ROTATION_FACTOR, interpolation='bilinear', fill_mode='nearest', fill_value=0.0),
    tf.keras.layers.RandomZoom(ZOOM_FACTOR, interpolation='bilinear', fill_mode='nearest', fill_value=0.0),
    tf.keras.layers.RandomContrast(CONTRAST_FACTOR),
    tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=BRIGHTNESS_DELTA)),
], name='data_augmentation_pipeline')

# === TensorFlow Dataset Creation and Preprocessing ===
def preprocess_and_augment(image, label, augmentation_pipeline=None, apply_noise=False):
    image = tf.cast(image, tf.float32)

    if augmentation_pipeline:
        image = augmentation_pipeline(image, training=True)

    if apply_noise and GAUSSIAN_STD > 0:
        image = image + tf.random.normal(tf.shape(image), mean=0.0, stddev=GAUSSIAN_STD)
        image = tf.clip_by_value(image, 0.0, 255.0)

    # MobileNetV2 preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000)
train_ds = train_ds.map(lambda x, y: preprocess_and_augment(x, y, data_augmentation, apply_noise=True),
                        num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.map(lambda x, y: preprocess_and_augment(x, y),
                    num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.map(lambda x, y: preprocess_and_augment(x, y),
                     num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# === Model Definition (MobileNetV2 Transfer Learning) ===
base_model = keras.applications.MobileNetV2 (
    input_shape=INPUT_SHAPE,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(DROPOUT),
    keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
])

# === Base Model Compilation ===
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Base Model Training ===
print("\n--- Training Base Model ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=BASE_EPOCHS,
    class_weight=class_weights,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)]
)

# === Fine-tuning Setup and Training ===
base_model.trainable = True
fine_tune_at = 100
print(f"Fine-tuning from layer {fine_tune_at} ({base_model.layers[fine_tune_at].name}) onwards.")
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.96
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

print("\n--- Fine-tuning Model ---")
fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    class_weight=class_weights
)

# === Model Evaluation ===
print("\n--- Evaluating Model on Test Set ---")
test_loss, test_acc = model.evaluate(test_ds)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# === Predictions and Confusion Matrix ===
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true_labels = []
for _, labels_batch in test_ds:
    y_true_labels.extend(labels_batch.numpy())
y_true_labels = np.array(y_true_labels)

min_len = min(len(y_pred), len(y_true_labels))
y_pred = y_pred[:min_len]
y_true_labels = y_true_labels[:min_len]

cm = confusion_matrix(y_true_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# === Training History Visualization ===
acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
loss = history.history['loss'] + fine_tune_history.history['loss']
val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.axvline(x=BASE_EPOCHS - 1, color='gray', linestyle='--', label='Fine-tuning Start')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.axvline(x=BASE_EPOCHS - 1, color='gray', linestyle='--', label='Fine-tuning Start')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()

# === Model Saving ===
# model.save('micropix.keras')

# === Results Export ===
class_weights_summary = {k: round(float(v), 3) for k, v in class_weights.items()}

augmentation_layers = [layer.name for layer in data_augmentation.layers]
augmentation_summary_str = ", ".join(augmentation_layers)

config = {
    'Image Size': IMAGE_SIZE,
    'Batch Size': BATCH_SIZE,
    'Number of Classes': num_classes,
    'Class Weights': class_weights_summary,
    'Base Epochs': BASE_EPOCHS,
    'Fine-tune Epochs': EPOCHS,
    'Data Augmentation': augmentation_summary_str,
    'RandomFlip Mode': FLIP_MODE,
    'RandomRotation Factor': ROTATION_FACTOR,
    'RandomZoom Factor': ZOOM_FACTOR,
    'RandomContrast Factor': CONTRAST_FACTOR,
    'RandomBrightness MaxDelta': BRIGHTNESS_DELTA,
    'Gaussian Noise STD': GAUSSIAN_STD,
    'Dropout': DROPOUT,
    'Train Size': len(X_train),
    'Validation Size': len(X_val),
    'Test Size': len(X_test),
    'Optimizer (Base)': 'Adam',
    'Optimizer (Fine-tune)': f'Adam with ExponentialDecay(initial_lr={1e-4}, decay_rate={0.98})',
    'Early Stopping Patience': early_stopping.patience,
    'Comments': 'Using MobileNetV2, fine tune more layers',
}

export_report(config, model, history, fine_tune_history, y_true_labels, y_pred, class_names, BASE_EPOCHS, test_acc, filename='micropix_report.pdf')
