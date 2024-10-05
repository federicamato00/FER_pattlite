import h5py
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
import matplotlib 
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from tensorflow.keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold

#### per creare nuove cartelle e salvare tutto ####

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

# Carica i migliori iperparametri dal file
with open('best_hyperparameters.txt', 'r') as f:
    best_hps = {}
    for line in f:
        name, value = line.strip().split(': ')
        if name in ['units']:
            best_hps[name] = int(float(value))
        else:
            best_hps[name] = float(value)

# Carica i migliori iperparametri per fine tuning dal file
with open('best_finetune_hyperparameters.txt', 'r') as f:
    best_hps_ft = {}
    for line in f:
        name, value = line.strip().split(': ')
        if name in ['units']:
            best_hps_ft[name] = int(float(value))
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
FT_LR_DECAY_RATE = 0.5

FT_ES_PATIENCE = 20
FT_DROPOUT = best_hps['train_dropout']
dropout_rate = best_hps['dropout_rate']

ES_LR_MIN_DELTA = 0.003

dataset_name='Bosphorus'

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

if 'CK+' in dataset_name:
    file_output = 'ckplus.h5'
elif 'RAFDB' in dataset_name:
    file_output = 'rafdb.h5'
elif 'FERP' in dataset_name:
    file_output = 'ferp.h5'
elif 'JAFFE' in dataset_name:
    file_output = 'jaffe.h5'
elif 'Bosphorus' in dataset_name:
    file_output = 'bosphorus_prova.h5'
elif 'BU_3DFE' in dataset_name:
    file_output = 'bu_3dfe.h5'
else:
    file_output = 'dataset.h5'

# Carica i dati dal file HDF5
with h5py.File(file_output, 'r') as f:
    X = np.array(f['X'])
    y = np.array(f['y'])

# Verifica che i dati non siano vuoti
assert X.size > 0, "X è vuoto"
assert y.size > 0, "y è vuoto"

# Shuffle dei dati
X, y = shuffle(X, y)

print("Shape of samples: {}".format(X.shape))
print("Shape of labels: {}".format(y.shape))

# Calcola i pesi delle classi
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(class_weights))

# Definisci il modello
def build_model():
    input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
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
        tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(best_hps['l2_reg']))
    ], name='patch_extraction')

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
    pre_classification = tf.keras.Sequential([
        tf.keras.layers.Dense(best_hps['units'], activation='relu', kernel_regularizer=l2(best_hps['l2_reg'])), 
        tf.keras.layers.BatchNormalization(),  
        tf.keras.layers.Dropout(dropout_rate)
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
    x = ExpandDimsLayer(axis=1)(x)
    x = self_attention([x, x])
    x = SqueezeLayer(axis=1)(x)
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs, name='train-head')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=TRAIN_LR, global_clipnorm=3.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Definisci la cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Esegui la cross-validation
val_accuracies = []
val_losses = []

for train_index, val_index in kf.split(X):
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]
    
    model = build_model()
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True)
    learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)
    
    history = model.fit(X_train_fold, y_train_fold, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_val_fold, y_val_fold), verbose=0, 
                        class_weight=class_weights, callbacks=[early_stopping_callback, learning_rate_callback])
    
    val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold)
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss)

# Calcola le metriche medie di validazione
mean_val_accuracy = np.mean(val_accuracies)
mean_val_loss = np.mean(val_losses)

print(f"Mean Validation Accuracy: {mean_val_accuracy}")
print(f"Mean Validation Loss: {mean_val_loss}")

# Addestra il modello finale su tutti i dati di training
model = build_model()
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True)
learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)
history = model.fit(X, y, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, validation_split=0.2, verbose=0, 
                    class_weight=class_weights, callbacks=[early_stopping_callback, learning_rate_callback])

# Continua con il fine-tuning e il salvataggio del modello come nel tuo codice originale