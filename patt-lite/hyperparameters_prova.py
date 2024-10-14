import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from keras_tuner import HyperModel, RandomSearch
import os
import h5py
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle

NUM_CLASSES = 7
IMG_SHAPE = (120, 120, 3)
dataset_name = 'Bosphorus'
# Caricamento dati
file_output = os.path.join('datasets', dataset_name, 'bosphorus_data_augmentation.h5')
best_path_save = dataset_name + '_hyperparameters'
best_path = os.path.join(best_path_save, 'bosphorus_data_augmentation_hyperparameters.txt')
if not os.path.exists(best_path_save):
    os.makedirs(best_path_save)

with h5py.File(file_output, 'r') as f:
    X_train = np.array(f['X_train'])
    y_train = np.array(f['y_train'])
    X_valid = np.array(f['X_val'])
    y_valid = np.array(f['y_val'])
    X_test = np.array(f['X_test'])
    y_test = np.array(f['y_test'])


X_train,y_train = shuffle(X_train,y_train,random_state=42)
X_valid,y_valid = shuffle(X_valid,y_valid,random_state=42)
X_test,y_test = shuffle(X_test,y_test,random_state=42)

class PattLiteHyperModel(HyperModel):
    def __init__(self, IMG_SHAPE, NUM_CLASSES):
        self.IMG_SHAPE = IMG_SHAPE
        self.NUM_CLASSES = NUM_CLASSES

    def build(self, hp):
        input_layer = tf.keras.Input(shape=self.IMG_SHAPE, name='universal_input')
        batch_size = hp.Choice('BATCH_SIZE', [8, 16, 32, 64])
        
        sample_resizing = tf.keras.layers.Resizing(224, 224, name="resize")
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(mode='horizontal'), 
            tf.keras.layers.RandomContrast(factor=0.3)], name="augmentation")
        
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input

        backbone = tf.keras.applications.mobilenet.MobileNet(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        backbone.trainable = False
        base_model = tf.keras.Model(backbone.input, backbone.layers[-29].output, name='base_model')

        self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')
        patch_extraction = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'),
        tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', 
                               kernel_regularizer=regularizers.l2(hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='LOG')))
        ], name='patch_extraction')
        

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        pre_classification = tf.keras.Sequential([
        tf.keras.layers.Dense(hp.Int('units', min_value=32, max_value=128, step=32), activation='relu', 
                              kernel_regularizer=regularizers.l2(hp.Float('l2_reg_2', min_value=1e-5, max_value=1e-2, sampling='LOG'))),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1))
        ], name='pre_classification')

        prediction_layer = tf.keras.layers.Dense(self.NUM_CLASSES, activation="softmax", name='classification_head')

        inputs = input_layer
        x = sample_resizing(inputs)
        x = data_augmentation(x)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = patch_extraction(x)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(hp.Float('TRAIN_DROPOUT', min_value=0.2, max_value=0.5, step=0.1))(x)
        x = pre_classification(x)
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        x = self_attention([x, x])
        x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(x)
        outputs = prediction_layer(x)
        
        model = tf.keras.Model(inputs, outputs, name='train-head')
        model.compile(optimizer=tf.keras.optimizers.Adam(
                          learning_rate=hp.Float('TRAIN_LR', min_value=1e-4, max_value=1e-2, sampling='LOG')), ######provare a mettere min_value a 1e-6
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        
        return model
    

class PattLiteFineTuneHyperModel(HyperModel):
    def __init__(self, IMG_SHAPE, NUM_CLASSES, base_model, unfreeze):
        self.IMG_SHAPE = IMG_SHAPE
        self.NUM_CLASSES = NUM_CLASSES
        self.base_model = base_model
        self.unfreeze = unfreeze

    def build(self, hp):
        self.base_model.trainable = True
        batch_size = hp.Choice('BATCH_SIZE_FT', [8, 16, 32, 64])
        fine_tune_from = len(self.base_model.layers) - self.unfreeze
        for layer in self.base_model.layers[:fine_tune_from]:
            layer.trainable = False
        for layer in self.base_model.layers[fine_tune_from:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        
        input_layer = tf.keras.Input(shape=self.IMG_SHAPE, name='universal_input')
        sample_resizing = tf.keras.layers.Resizing(224, 224, name="resize")
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        prediction_layer = tf.keras.layers.Dense(self.NUM_CLASSES, activation="softmax", name='classification_head')
        data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip(mode='horizontal'), 
                                        tf.keras.layers.RandomContrast(factor=0.3)], name="augmentation")
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input     
        patch_extraction = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'), 
        tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'), 
        tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(hp.Float('l2_reg_fine_tuning', min_value=1e-5, max_value=1e-2, sampling='LOG')))
    ], name='patch_extraction')
        
        pre_classification = tf.keras.Sequential([
        tf.keras.layers.Dense(hp.Int('units_ft', min_value=32, max_value=128, step=32), activation='relu', 
                              kernel_regularizer=regularizers.l2(hp.Float('l2_reg_FT', min_value=1e-5, max_value=1e-2, sampling='LOG'))),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(hp.Float('dropout_rate_FT', min_value=0.3, max_value=0.7, step=0.1))
        ], name='pre_classification')
        self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')
        inputs = input_layer
        x = sample_resizing(inputs)
        x = data_augmentation(x)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = patch_extraction(x)
        x = tf.keras.layers.SpatialDropout2D(hp.Float('FT_DROPOUT', min_value=0.2, max_value=0.5, step=0.1))(x)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(hp.Float('FT_DROPOUT', min_value=0.2, max_value=0.5, step=0.1))(x)
        x = pre_classification(x)
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
        x = self_attention([x, x])
        x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(x)
        x = tf.keras.layers.Dropout(hp.Float('FT_DROPOUT', min_value=0.2, max_value=0.5, step=0.1))(x)
        outputs = prediction_layer(x)

        model = tf.keras.Model(inputs, outputs, name='finetune-backbone')
        
        model.compile(optimizer=tf.keras.optimizers.Adam(
                          learning_rate=hp.Float('FT_LR', min_value=1e-6, max_value=1e-2, sampling='LOG')),
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        
        return model


backbone = tf.keras.applications.mobilenet.MobileNet(
input_shape=(224, 224, 3), include_top=False, weights='imagenet')
backbone.trainable = False
base_model = tf.keras.Model(backbone.input, backbone.layers[-29].output, name='base_model')

tuner_train = RandomSearch(
    PattLiteHyperModel(IMG_SHAPE=(120, 120, 3), NUM_CLASSES=7),
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=1,
    directory='pattlite_tuning',
    project_name='pattlite_train'
)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))


# Ricerca iperparametri per il modello di addestramento
tuner_train.search(X_train, y_train,
                   validation_data=(X_valid, y_valid),
                   epochs=10,
                   class_weight=class_weights,
                   )  # Usa il batch_size qui

best_train_model = tuner_train.get_best_models(num_models=1)[0]
best_train_hp = tuner_train.get_best_hyperparameters(num_trials=1)[0]

# Utilizzare il miglior modello trovato per il fine-tuning
tuner_finetune = RandomSearch(
    PattLiteFineTuneHyperModel(IMG_SHAPE=(120, 120, 3), NUM_CLASSES=7, base_model=best_train_model, unfreeze=59),
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=1,
    directory='pattlite_tuning',
    project_name='pattlite_finetune'
)

# Ricerca iperparametri per il modello di fine-tuning
tuner_finetune.search(X_train, y_train,
                      validation_data=(X_valid, y_valid),
                      epochs=10,
                      class_weight=class_weights,
                      )  # Usa il batch_size qui

best_finetune_model = tuner_finetune.get_best_models(num_models=1)[0]
best_finetune_hp = tuner_finetune.get_best_hyperparameters(num_trials=1)[0]

with open(best_path, 'w') as f:
    for param, value in best_train_hp.values.items():
        f.write(f"{param}: {value}\n")
    for param, value in best_finetune_hp.values.items():
        f.write(f"{param}: {value}\n")