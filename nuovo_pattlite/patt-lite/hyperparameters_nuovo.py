import os
import h5py
import keras
import keras_tuner as kt
import numpy as np
import tensorflow as tf
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

global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
pre_classification = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization()
], name='pre_classification')

prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name='classification_head')

dataset_name = 'CK+'

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

# Funzione per costruire il modello per la fase di addestramento iniziale
def build_initial_model(hp):
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_l2 = hp.Float('l2', min_value=1e-5, max_value=1e-2, sampling='log')
    inputs = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
    x = sample_resizing(inputs)
    x = data_augmentation(x)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'),
        tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(hp_l2))
    ], name='patch_extraction')(x)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(hp_dropout)(x)
    x = pre_classification(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = ExpandDimsLayer(axis=-1)(x)
    x = self_attention([x, x])
    x = SqueezeLayer(axis=-1)(x)
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs, name='train-head')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate, global_clipnorm=3.0),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Configura Keras Tuner per la fase di addestramento iniziale
tuner_initial = kt.RandomSearch(
    build_initial_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='initial_training'
)

seven_classes = dataset_name + '_numClasses7'
path_file = os.path.join('datasets', 'data_augmentation', seven_classes, 'ckplus_data_augmentation.h5')
# Carica le immagini e le etichette
X_train, y_train, X_valid, y_valid, X_test, y_test = load_images_and_labels(path_file)

# Ridimensiona le immagini difficili

print("Shape of train_sample: {}".format(X_train.shape))
print("Shape of train_label: {}".format(y_train.shape))
print("Shape of valid_sample: {}".format(X_valid.shape))
print("Shape of valid_label: {}".format(y_valid.shape))
print("Shape of test_sample: {}".format(X_test.shape))
print("Shape of test_label: {}".format(y_test.shape))

# Verifica che i dati non siano vuoti
assert X_train.size > 0, "X_train è vuoto"
assert y_train.size > 0, "y_train è vuoto"
assert X_valid.size > 0, "X_valid è vuoto"
assert y_valid.size > 0, "y_valid è vuoto"
assert X_test.size > 0, "X_test è vuoto"
assert y_test.size > 0, "y_test è vuoto"

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True)
# Esegui la ricerca degli iperparametri per la fase di addestramento iniziale
tuner_initial.search(X_train, y_train, epochs=15, validation_data=(X_valid, y_valid), callbacks=[early_stopping_callback])

# Ottieni i migliori iperparametri per la fase di addestramento iniziale
best_hps_initial = tuner_initial.get_best_hyperparameters(num_trials=1)[0]

print(f"""
I migliori iperparametri per la fase di addestramento iniziale sono:
- Dropout: {best_hps_initial.get('dropout')}
- Learning rate: {best_hps_initial.get('learning_rate')}
- L2 Regularizer: {best_hps_initial.get('l2')}
""")

# Ricostruisci il modello con i migliori iperparametri per la fase di addestramento iniziale
model_initial = tuner_initial.hypermodel.build(best_hps_initial)

# Addestra il modello con i migliori iperparametri per la fase di addestramento iniziale
history_initial = model_initial.fit(X_train, y_train, epochs=TRAIN_EPOCH, validation_data=(X_valid, y_valid), callbacks=[early_stopping_callback])

# Salva i pesi del modello addestrato nella fase iniziale
initial_weights_path = os.path.join("initial_weights", dataset_name, "initial_model.weights.h5")
os.makedirs(os.path.dirname(initial_weights_path), exist_ok=True)
model_initial.save_weights(initial_weights_path)

# Funzione per costruire il modello per la fase di fine-tuning
def build_finetune_model(hp):
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-5, 1e-4, 1e-3])
    hp_l2 = hp.Float('l2', min_value=1e-5, max_value=1e-2, sampling='log')

    base_model.trainable = True
    unfreeze = 59  # Sbloccato più livelli per il fine-tuning
    fine_tune_from = len(base_model.layers) - unfreeze
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_from:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inputs = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')

    x = sample_resizing(inputs)
    x = data_augmentation(x)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'),
        tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(hp_l2))
    ], name='patch_extraction')(x)
    x = global_average_layer(x)
    x = pre_classification(x)
    x = ExpandDimsLayer(axis=-1)(x)
    x = self_attention([x, x])
    x = SqueezeLayer(axis=-1)(x)
    x = tf.keras.layers.Dropout(hp_dropout)(x)
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs, name='finetune-backbone')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate, global_clipnorm=3.0),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Carica i pesi del modello addestrato nella fase iniziale
    model.load_weights(initial_weights_path)
    print(f"Pesi del modello iniziale caricati da: {initial_weights_path}")

    return model

# Configura Keras Tuner per la fase di fine-tuning
tuner_finetune = kt.RandomSearch(
    build_finetune_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='finetuning'
)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=FT_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True)

# Esegui la ricerca degli iperparametri per la fase di fine-tuning
tuner_finetune.search(X_train, y_train, epochs=15, validation_data=(X_valid, y_valid), callbacks=[early_stopping_callback])

# Ottieni i migliori iperparametri per la fase di fine-tuning
best_hps_finetune = tuner_finetune.get_best_hyperparameters(num_trials=1)[0]

print(f"""
I migliori iperparametri per la fase di fine-tuning sono:
- Dropout: {best_hps_finetune.get('dropout')}
- Learning rate: {best_hps_finetune.get('learning_rate')}
- L2 Regularizer: {best_hps_finetune.get('l2')}
""")

def save_parameters(params, directory, filename="parameters.txt"):
    """
    Salva i parametri in un file .txt nella directory specificata.
    """
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")


# Definisci i migliori iperparametri in un dizionario
best_hps_dict = {
    'dropout_initial': best_hps_initial.get('dropout'),
    'learning_rate_initial': best_hps_initial.get('learning_rate'),
    'l2_initial': best_hps_initial.get('l2'),
    'dropout': best_hps_finetune.get('dropout'),
    'learning_rate': best_hps_finetune.get('learning_rate'),
    'l2': best_hps_finetune.get('l2')
}

# Salva i migliori iperparametri su un file
save_parameters(best_hps_dict, directory='CK+_hyperparameters_numClasses7', filename=dataset_name+'_hyperparameters_numClasses7_new.txt')

# Carica i pesi migliori del modello iniziale nel modello di fine-tuning
model_finetune = tuner_finetune.hypermodel.build(best_hps_finetune)
model_finetune.load_weights(initial_weights_path)

# Continua con l'addestramento del fine-tuning
history_finetune = model_finetune.fit(X_train, y_train, epochs=FT_EPOCH, validation_data=(X_valid, y_valid), callbacks=[early_stopping_callback])

# Salva i pesi del modello fine-tuned
finetuned_weights_path = os.path.join("finetuned_weights", dataset_name, "finetuned_model.weights.h5")
os.makedirs(os.path.dirname(finetuned_weights_path), exist_ok=True)
model_finetune.save_weights(finetuned_weights_path)