import datetime
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import os
from tensorflow.keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from skimage import exposure, filters
from skimage.util import img_as_float
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import h5py
from skimage import exposure, filters, restoration, img_as_float
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Layer

def visualize_intermediate_steps(original, gray, equalized, blurred, normalized, processed):
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Gray Image')
    axes[1].axis('off')
    
    axes[2].imshow(equalized, cmap='gray')
    axes[2].set_title('CLAHE')
    axes[2].axis('off')

    axes[3].imshow(blurred, cmap='gray')
    axes[3].set_title('Gaussian Blur')
    axes[3].axis('off')

    axes[4].imshow(normalized, cmap='gray')
    axes[4].set_title('Normalized')
    axes[4].axis('off')

    # axes[5].imshow(edged_detection, cmap='gray')
    # axes[5].set_title('Thresholded')
    # axes[5].axis('off')

    axes[5].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    axes[5].set_title('Processed Image')
    axes[5].axis('off')

    plt.savefig('preprocessing_example_5.png')
    plt.show()

def convert_to_gray(image):
    return cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

def apply_clahe(gray_image, clip_limit=0.01):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray_image.copy())

def apply_gaussian_blur(equalized_image, sigma=1.0):
    return cv2.GaussianBlur(equalized_image.copy(), (5, 5), sigma)

def normalize_image(blurred_image):
    return cv2.normalize(blurred_image.copy(), None, 0, 255, cv2.NORM_MINMAX)

def apply_threshold(normalized_image, threshold=0.5):
    _, thresholded = cv2.threshold(normalized_image.copy(), int(threshold * 255), 255, cv2.THRESH_BINARY)
    return thresholded

def convert_to_bgr(thresholded_image):
    return cv2.cvtColor(thresholded_image.copy(), cv2.COLOR_GRAY2BGR)



def preprocess_image(image, clip_limit, sigma, return_intermediate=False):
    image_copy = image.copy()  # Assicurati di fare una copia dell'immagine originale
    gray = convert_to_gray(image_copy)
    equalized = apply_clahe(gray, clip_limit)
    blurred = apply_gaussian_blur(equalized, sigma)
    normalized = normalize_image(blurred)
    # edged_detection = edge_detection(normalized)
    # thresholded = apply_threshold(normalized, threshold)
    processed = convert_to_bgr(normalized)
    if return_intermediate:
        return gray, equalized, blurred, normalized,processed
    else: 
        return processed

def identify_difficult_images(model, images, labels):
    predictions = model.predict(images)
    difficult_images = []

    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if np.argmax(pred) != label:
            difficult_images.append(i)

    return np.array(difficult_images)

def find_best_params(model, X_train, y_train, X_valid, y_valid, X_test, y_test, param_grid):
    grid = ParameterGrid(param_grid)
    best_params = None
    best_accuracy = 0
    min_difficult_images = float('inf')

    for params in grid:
        processed_train_images = [preprocess_image(img, params['clip_limit'], params['sigma']) for img in X_train]
        processed_val_images = [preprocess_image(img, params['clip_limit'], params['sigma']) for img in X_valid]
        processed_test_images = [preprocess_image(img, params['clip_limit'], params['sigma']) for img in X_test]
        
        processed_train_images = np.array(processed_train_images).reshape(len(processed_train_images), 120, 120, 3)
        processed_val_images = np.array(processed_val_images).reshape(len(processed_val_images), 120, 120, 3)
        processed_test_images = np.array(processed_test_images).reshape(len(processed_test_images), 120, 120, 3)
        
        processed_train_images = tf.image.resize(processed_train_images, (120, 120))
        processed_val_images = tf.image.resize(processed_val_images, (120, 120))
        processed_test_images = tf.image.resize(processed_test_images, (120, 120))
        
        train_predictions = model.predict(processed_train_images)
        val_predictions = model.predict(processed_val_images)
        test_predictions = model.predict(processed_test_images)
        
        train_accuracy = accuracy_score(y_train, np.argmax(train_predictions, axis=1))
        val_accuracy = accuracy_score(y_valid, np.argmax(val_predictions, axis=1))
        test_accuracy = accuracy_score(y_test, np.argmax(test_predictions, axis=1))
        
        overall_accuracy = (train_accuracy + val_accuracy + test_accuracy) / 3
        
        difficult_train_images = identify_difficult_images(model, processed_train_images, y_train)
        difficult_val_images = identify_difficult_images(model, processed_val_images, y_valid)
        difficult_test_images = identify_difficult_images(model, processed_test_images, y_test)
        
        total_difficult_images = len(difficult_train_images) + len(difficult_val_images) + len(difficult_test_images)
        
        if overall_accuracy > best_accuracy or (overall_accuracy == best_accuracy and total_difficult_images < min_difficult_images):
            best_accuracy = overall_accuracy
            best_params = params
            min_difficult_images = total_difficult_images

    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy}")
    print(f"Minimum difficult images: {min_difficult_images}")

    return best_params, best_accuracy, min_difficult_images

def save_data(file_path, train_images, train_labels, val_images, val_labels, test_images, test_labels):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('X_train', data=train_images)
        f.create_dataset('y_train', data=train_labels)
        f.create_dataset('X_valid', data=val_images)
        f.create_dataset('y_valid', data=val_labels)
        f.create_dataset('X_test', data=test_images)
        f.create_dataset('y_test', data=test_labels)

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

# Carica i migliori iperparametri dal file
with open('best_hyperparameters.txt', 'r') as f:
    best_hps = {}
    for line in f:
        name, value = line.strip().split(': ')
        if name in ['units']:
            best_hps[name] = int(float(value))  # Converti esplicitamente a int
        else:
            best_hps[name] = float(value)

# Carica i migliori iperparametri per fine tuning dal file
with open('best_finetune_hyperparameters.txt', 'r') as f:
    best_hps_ft = {}
    for line in f:
        name, value = line.strip().split(': ')
        if name in ['units']:
            best_hps_ft[name] = int(float(value))  # Converti esplicitamente a int
        else:
            best_hps_ft[name] = float(value)

# Parametri
NUM_CLASSES = 7
IMG_SHAPE = (120, 120, 3)
BATCH_SIZE = 8

TRAIN_EPOCH = 100
TRAIN_LR = best_hps['learning_rate']
TRAIN_ES_PATIENCE = 5
TRAIN_LR_PATIENCE = 3
TRAIN_MIN_LR = 1e-6
TRAIN_DROPOUT = best_hps['train_dropout']

FT_EPOCH = 500
FT_LR = best_hps_ft['ft_learning_rate']
FT_LR_DECAY_STEP = 80.0
FT_LR_DECAY_RATE = 0.5 #era a 1 ma ho messo 0.5

FT_ES_PATIENCE = 20 #numero di epoche di tolleranza per l'arresto anticipato
FT_DROPOUT = best_hps['train_dropout']
dropout_rate = best_hps['dropout_rate']

ES_LR_MIN_DELTA = 0.003 #quantità minima di cambiamento per considerare un miglioramento

dataset_name='Bosphorus'

if 'CK+' in dataset_name:
    file_output = 'ckplus.h5'
elif 'RAFDB' in dataset_name:
    file_output = 'rafdb.h5'
elif 'FERP' in dataset_name:
    file_output = 'ferp.h5'
elif 'JAFFE' in dataset_name:
    file_output = 'jaffe.h5'
elif 'Bosphorus' in dataset_name:
    file_output = 'bosphorus_data_augmentation_5.h5'
elif 'BU_3DFE' in dataset_name:
    file_output = 'bu_3dfe.h5'
else:
    file_output = 'dataset.h5'

name_file_path = os.path.join('datasets', dataset_name, file_output)
# Supponiamo che i tuoi dati siano memorizzati in un file HDF5 chiamato 'data.h5'
with h5py.File(name_file_path, 'r') as f:
    X_train = np.array(f['X_train'])
    y_train = np.array(f['y_train'])
    X_valid = np.array(f['X_val'])
    y_valid = np.array(f['y_val'])
    X_test = np.array(f['X_test'])
    y_test = np.array(f['y_test'])

# Verifica che i dati non siano vuoti
assert X_train.size > 0, "X_train è vuoto"
assert y_train.size > 0, "y_train è vuoto"
assert X_valid.size > 0, "X_valid è vuoto"
assert y_valid.size > 0, "y_valid è vuoto"
assert X_test.size > 0, "X_test è vuoto"
assert y_test.size > 0, "y_test è vuoto"

# Supponiamo che X_train sia già definito e contenga le immagini
example_index = 0  # Indice dell'immagine da visualizzare
original_image = X_train[example_index].copy()

# Visualizza l'immagine utilizzando Matplotlib
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')  # Nasconde gli assi
plt.show()

# # Load your data here, PAtt-Lite was trained with h5py for shorter loading time
# X_train, y_train = shuffle(X_train, y_train)

print("Shape of train_sample: {}".format(X_train.shape))
print("Shape of train_label: {}".format(y_train.shape))
print("Shape of valid_sample: {}".format(X_valid.shape))
print("Shape of valid_label: {}".format(y_valid.shape))
print("Shape of test_sample: {}".format(X_test.shape))
print("Shape of test_label: {}".format(y_test.shape))

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
X_train = X_train.reshape(X_train.shape[0], 120, 120, 3)
X_valid = X_valid.reshape(X_valid.shape[0], 120, 120, 3)
X_test = X_test.reshape(X_test.shape[0], 120, 120, 3)
print("Shape of train_sample: {}".format(X_train.shape))
sample_resizing = tf.keras.layers.Resizing(224, 224, name="resize")
data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode='horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ], name="augmentation")
    
preprocess_input = tf.keras.applications.mobilenet.preprocess_input

backbone = tf.keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
backbone.trainable = False
base_model = tf.keras.Model(backbone.input, backbone.layers[-29].output, name='base_model')

self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')

patch_extraction = tf.keras.Sequential([
    tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'), 
    tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'), 
    tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(best_hps['l2_reg']))
], name='patch_extraction')

global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
pre_classification = tf.keras.Sequential([
    tf.keras.layers.Dense(best_hps['units'], activation='relu', kernel_regularizer=l2(best_hps['l2_reg'])), 
    tf.keras.layers.BatchNormalization(),  
    tf.keras.layers.Dropout(dropout_rate)  # Aggiungi dropout
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
x = ExpandDimsLayer(axis=1)(x)  # Aggiungi una dimensione di sequenza 
x = self_attention([x, x])
x = SqueezeLayer(axis=1)(x)  # Rimuovi la dimensione di sequenza dopo l'attenzione
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs, name='train-head')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=TRAIN_LR, global_clipnorm=3.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True)
learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)
history = model.fit(X_train, y_train, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_valid, y_valid), verbose=0, 
                    class_weight=class_weights, callbacks=[early_stopping_callback, learning_rate_callback])
test_loss, test_acc = model.evaluate(X_test, y_test)

predictions = model.predict(X_test)
difficult_images = []

difficult_train_images = identify_difficult_images(model, X_train, y_train)
difficult_val_images = identify_difficult_images(model, X_valid, y_valid)
difficult_test_images = identify_difficult_images(model, X_test, y_test)

# Esempio di utilizzo
param_grid = {
    'clip_limit': [0.007,0.008, 0.01, 0.02, 0.03],
    'sigma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    
}

best_params, best_accuracy, min_difficult_images = find_best_params(model, X_train, y_train, X_valid, y_valid, X_test, y_test, param_grid)

clip_limit, sigma = best_params['clip_limit'], best_params['sigma']

# Preprocessa tutte le immagini nei set di addestramento, validazione e test
processed_train_images = [preprocess_image(img, clip_limit, sigma) for img in X_train]
processed_val_images = [preprocess_image(img, clip_limit, sigma) for img in X_valid]
processed_test_images = [preprocess_image(img, clip_limit, sigma) for img in X_test]

# Salva tutte le immagini preprocessate
save_data('processed_bosphorus_5.h5', processed_train_images, y_train, 
          processed_val_images, y_valid, 
          processed_test_images, y_test)

# Visualizza un esempio di immagine processata con tutti gli step intermedi
example_index = 0  # Indice dell'immagine da visualizzare
original_image = X_train[example_index].copy()  # Assicurati di fare una copia dell'immagine originale
gray, equalized, blurred, normalized, processed_image = preprocess_image(original_image, clip_limit, sigma, return_intermediate=True)

visualize_intermediate_steps(original_image, gray, equalized, blurred, normalized, processed_image)