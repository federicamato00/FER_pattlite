import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
import keras_tuner as kt
from tensorflow.keras.regularizers import l2

# Parametri
NUM_CLASSES = 8
IMG_SHAPE = (120, 120, 3)

# Caricamento dati

### PER CK+ ### 
file_output = 'ckplus_data_augmentation_5.h5'
# file_output = 'ckplus_data_augmentation_1.h5'
# file_output = 'ckplus_data_augmentation_2.h5'
# file_output = 'ckplus_data_augmentation_3.h5'
# file_output = 'ckplus.h5'

### PER BOSPHORUS ###
# file_output = 'bosphorus.h5'
# file_output = 'bosphorus_data_augmentation_2.h5'
# file_output = 'bosphorus_data_augmentation_3.h5'
# file_output = 'bosphorus_data_augmentation_4.h5'
# file_output = 'bosphorus_data_augmentation_5.h5'

### PER BU-3DFE ###
# file_output = 'bu3dfe.h5'
# file_output = 'bu3dfe_data_augmentation_2.h5'
# file_output = 'bu3dfe_data_augmentation_3.h5'
# file_output = 'bu3dfe_data_augmentation_4.h5'
# file_output = 'bu3dfe_data_augmentation_5.h5'

with h5py.File(file_output, 'r') as f:
    X_train = np.array(f['X_train'])
    y_train = np.array(f['y_train'])
    X_valid = np.array(f['X_val'])
    y_valid = np.array(f['y_val'])
    X_test = np.array(f['X_test'])
    y_test = np.array(f['y_test'])

X_train, y_train = shuffle(X_train, y_train)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

def build_model(hp):
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
        tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='LOG')))
    ], name='patch_extraction')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
    pre_classification = tf.keras.Sequential([
        tf.keras.layers.Dense(hp.Int('units', min_value=32, max_value=128, step=32), activation='relu', kernel_regularizer=l2(hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='LOG'))),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1))
    ], name='pre_classification')

    prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name='classification_head')

    inputs = input_layer
    x = sample_resizing(inputs)
    x = data_augmentation(x)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = patch_extraction(x)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(hp.Float('train_dropout', min_value=0.2, max_value=0.5, step=0.1))(x)
    x = pre_classification(x)
    x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
    x = self_attention([x, x])
    x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(x)
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs, name='train-head')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = kt.Hyperband(build_model,
                     objective=kt.Objective('val_accuracy', direction='max'),
                     max_epochs=100,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-6)

# Aggiungi il batch size come iperparametro
tuner.search(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[stop_early, reduce_lr], batch_size=kt.HyperParameters().Int('batch_size', min_value=8, max_value=64, step=8))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


print(f"""
The hyperparameter search is complete. The optimal number of units in the dense layer is {best_hps.get('units')},
the optimal dropout rate is {best_hps.get('dropout_rate')},
the optimal train dropout is {best_hps.get('train_dropout')},
the optimal learning rate is {best_hps.get('learning_rate')},
the optimal L2 regularization is {best_hps.get('l2_reg')}.
""")

# Salva i migliori iperparametri in un file
with open('best_hyperparameters_ckplus.txt', 'w') as f:
    f.write(f"units: {best_hps.get('units')}\n")
    f.write(f"dropout_rate: {best_hps.get('dropout_rate')}\n")
    f.write(f"train_dropout: {best_hps.get('train_dropout')}\n")
    f.write(f"learning_rate: {best_hps.get('learning_rate')}\n")
    f.write(f"l2_reg: {best_hps.get('l2_reg')}\n")


