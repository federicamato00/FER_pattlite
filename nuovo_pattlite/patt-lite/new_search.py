import optuna
import tensorflow as tf
from tensorflow import keras
import h5py
import numpy as np
import os
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2

NUM_CLASSES = 7
IMG_SHAPE = (120, 120, 3)
BATCH_SIZE = 8

TRAIN_EPOCH = 100
TRAIN_ES_PATIENCE = 5
ES_LR_MIN_DELTA = 0.003

FT_EPOCH = 500
FT_ES_PATIENCE = 20
dataset_name = 'CK+'
# Funzione per caricare le immagini e le etichette
def load_images_and_labels(file_path):
    with h5py.File(file_path, 'r') as f:
        if file_path == 'bosphorus.h5':
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

seven_classes = dataset_name + '_numClasses7'
path_file = os.path.join('datasets', 'data_augmentation', seven_classes, 'ckplus_data_augmentation.h5')
# Carica le immagini e le etichette
X_train, y_train, X_valid, y_valid, X_test, y_test = load_images_and_labels(path_file)

@tf.keras.utils.register_keras_serializable()
class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

@tf.keras.utils.register_keras_serializable()
class SqueezeLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)


# Definisci la funzione obiettivo per Optuna
def objective(trial):
    # Definisci gli iperparametri da ottimizzare
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    num_units = trial.suggest_int('num_units', 32, 256, step=32)
    
    # Costruisci il modello
    input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
    sample_resizing = tf.keras.layers.Resizing(224, 224, name="resize")
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode='horizontal'),
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
        tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu')
    ], name='patch_extraction')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
    pre_classification = tf.keras.Sequential([
        tf.keras.layers.Dense(num_units, activation='relu'),
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
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = pre_classification(x)
    x = ExpandDimsLayer(axis=-1)(x)
    x = self_attention([x, x])
    x = SqueezeLayer(axis=-1)(x)
    outputs = prediction_layer(x)
    
    model = tf.keras.Model(inputs, outputs, name='train-head')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=3.0), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Addestra il modello
    history = model.fit(X_train, y_train, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_valid, y_valid), verbose=0)
    
    # Fine-tuning
    unfreeze = 59
    base_model.trainable = True
    fine_tune_from = len(base_model.layers) - unfreeze
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_from:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    x = sample_resizing(inputs)
    x = data_augmentation(x)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = patch_extraction(x)
    x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = pre_classification(x)
    x = ExpandDimsLayer(axis=-1)(x)
    x = self_attention([x, x])
    x = SqueezeLayer(axis=-1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = prediction_layer(x)
    
    model = tf.keras.Model(inputs, outputs, name='finetune-backbone')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=3.0), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Addestra il modello fine-tuned
    history_finetune = model.fit(X_train, y_train, epochs=FT_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_valid, y_valid), verbose=0)
    
    # Restituisci l'accuratezza di validazione
    val_accuracy = history_finetune.history['val_accuracy'][-1]
    return val_accuracy

# Crea uno studio Optuna e ottimizza
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Stampa i migliori iperparametri
print(f"Best trial: {study.best_trial.value}")
print(f"Best params: {study.best_trial.params}")
