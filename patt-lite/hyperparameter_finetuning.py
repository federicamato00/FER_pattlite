import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from tensorflow.keras.regularizers import l2

# Parametri
NUM_CLASSES = 7
IMG_SHAPE = (120, 120, 3)

# Caricamento dati
file_output = 'bosphorus_prova.h5'
with h5py.File(file_output, 'r') as f:
    X_train = np.array(f['X_train'])
    y_train = np.array(f['y_train'])
    X_valid = np.array(f['X_val'])
    y_valid = np.array(f['y_val'])
    X_test = np.array(f['X_test'])
    y_test = np.array(f['y_test'])

# Carica i migliori iperparametri trovati nella fase di addestramento iniziale
with open('best_hyperparameters.txt', 'r') as f:
    best_hps = {}
    for line in f:
        name, value = line.strip().split(': ')
        if name in ['units']:
            best_hps[name] = int(float(value))  # Converti esplicitamente a int
        else:
            best_hps[name] = float(value)


def build_finetune_model(hp):
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
        tf.keras.layers.Dropout(best_hps['dropout_rate'])
    ], name='pre_classification')

    prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name='classification_head')

    inputs = input_layer
    x = sample_resizing(inputs)
    x = data_augmentation(x)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = patch_extraction(x)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(best_hps['train_dropout'])(x)
    x = pre_classification(x)
    x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
    x = self_attention([x, x])
    x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(x)
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs, name='train-head')

    print("\nFinetuning ...")
    unfreeze = 59  # Sbloccato più livelli per il fine-tuning
    base_model.trainable = True
    fine_tune_from = len(base_model.layers) - unfreeze
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_from:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Float('ft_learning_rate', min_value=1e-6, max_value=1e-4, sampling='LOG', default=1e-5)),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Crea un nuovo tuner per il fine-tuning
ft_tuner = kt.Hyperband(build_finetune_model,
                        objective=kt.Objective('val_accuracy', direction='max'),
                        max_epochs=100,
                        factor=3,
                        directory='my_dir',
                        project_name='finetune_kt')

ft_stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
ft_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-6)

# Cerca i migliori iperparametri per il fine-tuning
ft_tuner.search(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[ft_stop_early, ft_reduce_lr])

best_ft_hps = ft_tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The fine-tuning hyperparameter search is complete. The optimal fine-tuning learning rate is {best_ft_hps.get('ft_learning_rate')}.
""")

# Salva i migliori iperparametri di fine-tuning in un file
with open('best_finetune_hyperparameters.txt', 'w') as f:
    f.write(f"ft_learning_rate: {best_ft_hps.get('ft_learning_rate')}\n")

