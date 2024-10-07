import h5py
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from tensorflow.keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Funzioni di utilità per creare directory e salvare parametri
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

dataset_name = 'Bosphorus'

class ExpandDimsLayer(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

class SqueezeLayer(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

# Determina il file di output in base al dataset
if 'CK+' in dataset_name:
    file_output = 'ckplus.h5'
elif 'RAFDB' in dataset_name:
    file_output = 'rafdb.h5'
elif 'FERP' in dataset_name:
    file_output = 'ferp.h5'
elif 'JAFFE' in dataset_name:
    file_output = 'jaffe.h5'
elif 'Bosphorus' in dataset_name:
    file_output = 'bosphorus_SMOTE.h5'
elif 'BU_3DFE' in dataset_name:
    file_output = 'bu_3dfe.h5'
else:
    file_output = 'dataset.h5'

# Carica i dati
name_file_path = os.path.join('datasets', dataset_name, file_output)
with h5py.File(name_file_path, 'r') as f:
    X_train = np.array(f['X_train'])
    y_train = np.array(f['y_train'])
    X_valid = np.array(f['X_val'])
    y_valid = np.array(f['y_val'])
    X_test = np.array(f['X_test'])
    y_test = np.array(f['y_test'])

assert X_train.size > 0, "X_train è vuoto"
assert y_train.size > 0, "y_train è vuoto"
assert X_valid.size > 0, "X_valid è vuoto"
assert y_valid.size > 0, "y_valid è vuoto"
assert X_test.size > 0, "X_test è vuoto"
assert y_test.size > 0, "y_test è vuoto"

X_train, y_train = shuffle(X_train, y_train)

print("Shape of train_sample: {}".format(X_train.shape))
print("Shape of train_label: {}".format(y_train.shape))
print("Shape of valid_sample: {}".format(X_valid.shape))
print("Shape of valid_label: {}".format(y_valid.shape))
print("Shape of test_sample: {}".format(X_test.shape))
print("Shape of test_label: {}".format(y_test.shape))

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Preprocessamento e data augmentation
input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
X_train = X_train.reshape(X_train.shape[0], 120, 120, 3)
X_valid = X_valid.reshape(X_valid.shape[0], 120, 120, 3)
X_test = X_test.reshape(X_test.shape[0], 120, 120, 3)

sample_resizing = tf.keras.layers.Resizing(224, 224, name="resize")
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode='horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast(factor=0.3)
], name="augmentation")

preprocess_input = tf.keras.applications.mobilenet.preprocess_input

# Definisci il modello di base
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

# Costruisci il modello iniziale
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

initial_model = tf.keras.Model(inputs, outputs, name='initial_model')
initial_model.compile(optimizer=keras.optimizers.Adam(learning_rate=TRAIN_LR, global_clipnorm=3.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Addestra il modello iniziale
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True)
learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)
initial_model.fit(X_train, y_train, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_valid, y_valid), verbose=0, 
                  class_weight=class_weights, callbacks=[early_stopping_callback, learning_rate_callback])

# Funzione per classificare le immagini difficili
def classify_images(model, X, threshold=0.7):
    y_pred_prob = model.predict(X)
    y_pred = np.argmax(y_pred_prob, axis=1)
    conf_scores = np.max(y_pred_prob, axis=1)
    easy_indices = np.where(conf_scores >= threshold)[0]
    hard_indices = np.where(conf_scores < threshold)[0]
    return y_pred, easy_indices, hard_indices

# Classifica le immagini di test
y_pred, easy_indices, hard_indices = classify_images(initial_model, X_test)

# Processa le immagini difficili con LDA
lda = LDA(n_components=NUM_CLASSES-1)
X_train_lda = lda.fit_transform(X_train.reshape(X_train.shape[0], -1), y_train)
X_test_hard_lda = lda.transform(X_test[hard_indices].reshape(len(hard_indices), -1))

# Addestra un modello semplice sulle caratteristiche LDA
simple_model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(X_train_lda.shape[1],)),
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
simple_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
simple_model.fit(X_train_lda, y_train, epochs=10, batch_size=BATCH_SIZE)

# Valuta il modello semplice sulle immagini difficili
simple_model.evaluate(X_test_hard_lda, y_test[hard_indices])

# Combina le caratteristiche LDA con le caratteristiche originali
X_train_combined = np.concatenate([X_train.reshape(X_train.shape[0], -1), X_train_lda], axis=1)
X_test_combined = np.concatenate([X_test.reshape(X_test.shape[0], -1), X_test_hard_lda], axis=1)

# Costruisci il modello di addestramento
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

# Addestramento del modello
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True)
learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)
history = model.fit(X_train_combined, y_train, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_valid, y_valid), verbose=0, 
                    class_weight=class_weights, callbacks=[early_stopping_callback, learning_rate_callback])
test_loss, test_acc = model.evaluate(X_test_combined, y_test)

print(f"Test Accuracy: {test_acc}")

# Fine-tuning del modello
print("\nFinetuning ...")
unfreeze = 59
base_model.trainable = True
fine_tune_from = len(base_model.layers) - unfreeze
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_from:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

inputs = input_layer
x = sample_resizing(inputs)
x = data_augmentation(x)
x = preprocess_input(x)
x = base_model(x, training=False)
x = patch_extraction(x)
x = tf.keras.layers.SpatialDropout2D(FT_DROPOUT)(x)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(FT_DROPOUT)(x)
x = pre_classification(x)
x = ExpandDimsLayer(axis=1)(x)
x = self_attention([x, x])
x = SqueezeLayer(axis=1)(x)
x = tf.keras.layers.Dropout(FT_DROPOUT)(x)

outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs, name='finetune-backbone')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FT_LR, global_clipnorm=3.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def schedule(epoch, lr):
    return 0.5 * (1 + np.cos(np.pi * epoch / FT_EPOCH)) * FT_LR

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=ES_LR_MIN_DELTA, patience=FT_ES_PATIENCE, restore_best_weights=True)
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule=schedule)

checkpoint_dir = os.path.join("checkpoints/LDA_2", dataset_name)

checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir,'model_weights_epoch_{epoch:02d}.weights.h5'),
    save_weights_only=True,
    save_best_only=True,
)

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    model.load_weights(latest_checkpoint)
    print(f"Pesi del modello caricati da: {latest_checkpoint}")

history_finetune = model.fit(
    X_train_combined, y_train, 
    epochs=FT_EPOCH, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_valid, y_valid), 
    verbose=1, 
    initial_epoch=history.epoch[-TRAIN_ES_PATIENCE], 
    callbacks=[early_stopping_callback, scheduler_callback, tensorboard_callback, checkpoint_callback]
)

test_loss, test_acc = model.evaluate(X_test_combined, y_test)

# Salva il modello finale
final_model_dir = os.path.join("final_models/LDA_2", dataset_name)
base_dir = final_model_dir
unique_dir = create_unique_directory(base_dir)

model_name = os.path.join(unique_dir, f"{dataset_name}_model.keras")
model.save(model_name)

print(f"Modello salvato in: {model_name}")

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Valutazione del modello
y_pred_prob = model.predict(X_test_combined)
y_pred = np.argmax(y_pred_prob, axis=1)

correct_predictions = np.sum(y_pred == y_test)
incorrect_predictions = np.sum(y_pred != y_test)

print(f"Numero di predizioni corrette: {correct_predictions}")
print(f"Numero di predizioni sbagliate: {incorrect_predictions}")

accuracy = correct_predictions / len(y_test)
print(f"Accuratezza calcolata manualmente: {accuracy*100}%")

results_dir = os.path.join("results/LDA_2", dataset_name)

# Matrice di confusione
cm = confusion_matrix(y_test, y_pred)

if 'RAFDB' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
elif 'FERP' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
elif 'JAFFE' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
elif 'Bosphorus' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise','neutral']
elif 'CK+' in dataset_name:
    classNames = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
else:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classNames)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

plt.xticks(rotation=25, ha='right')
base_dir = results_dir
unique_dir = create_unique_directory(base_dir)

plt.savefig(os.path.join(unique_dir, 'confusion_matrix.png'))
plt.close()

# Grafici di accuratezza e perdita
history = history_finetune.history

train_accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
train_loss = history['loss']
val_loss = history['val_loss']

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.savefig(os.path.join(unique_dir, 'training_validation_plots.png'))
plt.show()

# Salva i parametri
params = {
    "NUM_CLASSES": NUM_CLASSES,
    "IMG_SHAPE": IMG_SHAPE,
    "BATCH_SIZE": BATCH_SIZE,
    "TRAIN_EPOCH": TRAIN_EPOCH,
    "TRAIN_LR": TRAIN_LR,
    "TRAIN_ES_PATIENCE": TRAIN_ES_PATIENCE,
    "TRAIN_LR_PATIENCE": TRAIN_LR_PATIENCE,
    "TRAIN_MIN_LR": TRAIN_MIN_LR,
    "TRAIN_DROPOUT": TRAIN_DROPOUT,
    "FT_EPOCH": FT_EPOCH,
    "FT_LR": FT_LR,
    "FT_LR_DECAY_STEP": FT_LR_DECAY_STEP,
    "FT_LR_DECAY_RATE": FT_LR_DECAY_RATE,
    "FT_ES_PATIENCE": FT_ES_PATIENCE,
    "FT_DROPOUT": FT_DROPOUT,
    "dropout_rate": dropout_rate,
    "ES_LR_MIN_DELTA": ES_LR_MIN_DELTA,
    "pre_classification": pre_classification.get_config(),
    "patch_extraction": patch_extraction.get_config(),
    "accuracy test set": test_acc,
    "accuracy train set": train_accuracy[-1],
    "accuracy validation set": val_accuracy[-1],
}

save_parameters(params, unique_dir)
print(f"Directory creata: {unique_dir}")
print(f"Parametri salvati in: {os.path.join(unique_dir, 'parameters.txt')}")