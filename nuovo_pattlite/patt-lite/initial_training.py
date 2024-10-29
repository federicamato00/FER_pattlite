import datetime
import keras
import numpy as np
import h5py
from sklearn.utils import compute_class_weight, shuffle
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from tensorflow.keras.layers import Layer
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pickle

class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LearningRateLogger, self).__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.get_config()['learning_rate']
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(epoch)
        self.learning_rates.append(tf.keras.backend.get_value(lr))

def plot_class_distribution(y_train, y_val, y_test, class_names):
    train_counts = np.bincount(y_train)
    val_counts = np.bincount(y_val)
    test_counts = np.bincount(y_test)
    print(f"Train counts: {train_counts}")
    print(f"Validation counts: {val_counts}")
    print(f"Test counts: {test_counts}")
    classes = np.arange(len(class_names))
    bar_width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(classes - bar_width, train_counts, width=bar_width, label='Train')
    ax.bar(classes, val_counts, width=bar_width, label='Validation')
    ax.bar(classes + bar_width, test_counts, width=bar_width, label='Test')
    ax.set_xlabel('Classi')
    ax.set_ylabel('Numero di campioni')
    ax.set_title('Distribuzione delle classi nei set di training, validazione e test')
    ax.set_xticks(classes)
    ax.set_xticklabels(class_names)
    ax.legend()
    plt.savefig('class_distribution_per_set.png')
    plt.show()

def create_unique_directory(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return base_dir
    counter = 1
    new_dir = f"{base_dir}_{counter}"
    while os.path.exists(new_dir):
        counter += 1
        new_dir = f"{base_dir}_{counter}"
    os.makedirs(new_dir)
    return new_dir

def save_parameters(params, directory, filename="parameters.txt"):
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

dataset_name='CK+'

class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

class SqueezeLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

best_path = dataset_name  + '_hyperparameters_numClasses7'
if os.path.exists(best_path):
    print(f"La directory '{best_path}' esiste.")
else:
    print(f"La directory '{best_path}' non esiste.")
    os.makedirs(best_path)

best_file = 'CK+_hyperparameters_numClasses7/CK+_hyperparameters_numClasses7_new.txt'
file_best_path = os.path.join(best_path, best_file)

with open(best_file, 'r') as f:
    best_hp = {}
    for line in f:
        print(line)
        name, value = line.strip().split(': ')
        if name in ['units']:
            best_hp[name] = int(float(value))
        else:
            best_hp[name] = float(value)

NUM_CLASSES = 7
IMG_SHAPE = (120, 120, 3)
BATCH_SIZE = 8

TRAIN_EPOCH = 100
TRAIN_LR = 1e-3
TRAIN_ES_PATIENCE = 5
TRAIN_LR_PATIENCE = 3
TRAIN_MIN_LR = 1e-6
TRAIN_DROPOUT = 0.1
TRAIN_DROPOUT = float(best_hp['dropout_initial'])
TRAIN_LR = float(best_hp['learning_rate_initial'])
l2_initial = float(best_hp['l2_initial'])

def load_images_and_labels(file_path):
    with h5py.File(file_path, 'r') as f:
        if file_path=='bosphorus.h5':
            X_train = np.array(f['X_train'])
            y_train = np.array(f['y_train'])
            X_test = np.array(f['X_test'])
            y_test = np.array(f['y_test'])
            X_valid = np.array(f['X_valid'])
            y_valid = np.array(f['y_valid'])
        else:
            X_train = np.array(f['X_train'])
            y_train = np.array(f['y_train'])
            X_valid = np.array(f['X_val'])
            y_valid = np.array(f['y_val'])
            X_test = np.array(f['X_test'])
            y_test = np.array(f['y_test'])
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def resize_images(X, target_size=(120, 120)):
    return np.array([tf.image.resize(image, target_size).numpy() for image in X])

seven_classes = dataset_name + '_numClasses7'
path_file = os.path.join('datasets', 'data_augmentation',seven_classes,'ckplus_data_augmentation.h5')
X_train, y_train , X_valid, y_valid, X_test, y_test= load_images_and_labels( path_file)

print("Shape of train_sample: {}".format(X_train.shape))
print("Shape of train_label: {}".format(y_train.shape))
print("Shape of valid_sample: {}".format(X_valid.shape))
print("Shape of valid_label: {}".format(y_valid.shape))
print("Shape of test_sample: {}".format(X_test.shape))
print("Shape of test_label: {}".format(y_test.shape))

assert X_train.size > 0, "X_train è vuoto"
assert y_train.size > 0, "y_train è vuoto"
assert X_valid.size > 0, "X_valid è vuoto"
assert y_valid.size > 0, "y_valid è vuoto"
assert X_test.size > 0, "X_test è vuoto"
assert y_test.size > 0, "y_test è vuoto"

if 'RAFDB' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
elif 'FERP' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
elif 'JAFFE' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
elif 'Bosphorus' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise','neutral']
elif 'CK+' in dataset_name:
    classNames = ['neutral', 'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
else:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

plot_class_distribution(y_train, y_valid, y_test, classNames)

X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)
X_test, y_test = shuffle(X_test, y_test)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
X_train = resize_images(X_train, target_size=(120, 120))
X_valid = resize_images(X_valid, target_size=(120, 120))
X_test = resize_images(X_test, target_size=(120, 120))

print("Shape of train_sample: {}".format(X_train.shape))
sample_resizing = tf.keras.layers.Resizing(224, 224, name="resize")
data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode='horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(factor=0.3)
    ], name="augmentation")

preprocess_input = tf.keras.applications.mobilenet.preprocess_input

backbone = tf.keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
backbone.trainable = False
base_model = tf.keras.Model(backbone.input, backbone.layers[-29].output, name='base_model')

self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')

patch_extraction = tf.keras.Sequential([
    tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'),
    tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(l2_initial))
], name='patch_extraction')

global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
pre_classification = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer = l2(l2_initial)),
    tf.keras.layers.BatchNormalization()
], name='pre_classification')

prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name='classification_head')

inputs = input_layer
x = sample_resizing(inputs)
x = data_augmentation(x)
x = preprocess_input(x)
x = base_model(x, training=False)
x = patch_extraction(x)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(TRAIN_DROPOUT)(x)
x = pre_classification(x)
x = ExpandDimsLayer(axis=-1)(x)
x = self_attention([x, x])
x = SqueezeLayer(axis=-1)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs, name='train-head')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=TRAIN_LR, global_clipnorm=3.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=0.003, restore_best_weights=True)
learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=0.003, min_lr=TRAIN_MIN_LR)

# Callback per salvare i pesi del modello
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join("initial_weights", dataset_name, "initial_model.weights.h5"),
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# Callback per salvare la cronologia dell'addestramento
history_path = os.path.join("initial_weights", dataset_name, "training_history.pkl")

def save_history_callback(epoch, logs):
    with open(history_path, 'wb') as file_pi:
        pickle.dump(model.history.history, file_pi)

history_callback = LambdaCallback(on_epoch_end=save_history_callback)

# Carica i pesi del modello salvati se esistono
initial_weights_path = os.path.join("initial_weights", dataset_name, "initial_model.weights.h5")
if os.path.exists(initial_weights_path):
    model.load_weights(initial_weights_path)
    print(f"Pesi del modello caricati da: {initial_weights_path}")

# Carica la cronologia dell'addestramento salvata se esiste
initial_epoch = 0
if os.path.exists(history_path):
    with open(history_path, 'rb') as file_pi:
        loaded_history = pickle.load(file_pi)
        initial_epoch = len(loaded_history['loss'])
    print(f"Cronologia dell'addestramento caricata da: {history_path}")

history = model.fit(X_train, y_train, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_valid, y_valid), verbose=0,
                    class_weight=class_weights, callbacks=[early_stopping_callback, learning_rate_callback, checkpoint_callback, history_callback],
                    initial_epoch=initial_epoch)

test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_acc}")

print(f"Pesi del modello iniziale salvati in: {os.path.join('initial_weights', dataset_name, 'initial_model.weights.h5')}")
print(f"Cronologia dell'addestramento salvata in: {history_path}")

# Carica la cronologia dell'addestramento
with open(history_path, 'rb') as file_pi:
    loaded_history = pickle.load(file_pi)

print(f"Cronologia dell'addestramento caricata: {loaded_history}")

# Visualizza e salva i grafici di val_loss, train_loss e accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loaded_history['loss'], label='Train Loss')
plt.plot(loaded_history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss')
plt.savefig('train_val_loss.png')

plt.subplot(1, 2, 2)
plt.plot(loaded_history['accuracy'], label='Train Accuracy')
plt.plot(loaded_history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Validation Accuracy')
plt.savefig('train_val_accuracy.png')

plt.show()

# Salva i valori di test accuracy e test loss
with open('test_results.txt', 'w') as f:
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_acc}\n")