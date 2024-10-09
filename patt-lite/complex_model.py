import numpy as np
import h5py
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Layer
import datetime

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
FT_DROPOUT = best_hps['train_dropout']
dropout_rate = FT_DROPOUT
FT_ES_PATIENCE = 20
ES_LR_MIN_DELTA = 0.003

dataset_name = 'Bosphorus'

# Funzione per caricare le immagini e le etichette
def load_images_and_labels(file_path):
    with h5py.File(file_path, 'r') as f:
        X = np.array(f['X'])
        y = np.array(f['y'])
    return X, y

# Funzione per ridimensionare le immagini
def resize_images(X, target_size=(224, 224)):
    return np.array([tf.image.resize(image, target_size).numpy() for image in X])

# Carica le immagini facili
path_easy = 'datasets/preprocessing/Bosphorus/SMOTE'

# Carica le immagini facili
X_easy_train, y_easy_train = load_images_and_labels(os.path.join(path_easy, 'easy_images.h5'))
X_easy_valid, y_easy_valid = load_images_and_labels(os.path.join(path_easy, 'easy_images_valid.h5'))
X_easy_test, y_easy_test = load_images_and_labels(os.path.join(path_easy, 'easy_images_test.h5'))

# Carica le immagini difficili
X_hard_train, y_hard_train = load_images_and_labels(os.path.join(path_easy, 'hard_images.h5'))
X_hard_valid, y_hard_valid = load_images_and_labels(os.path.join(path_easy, 'hard_images_valid.h5'))
X_hard_test, y_hard_test = load_images_and_labels(os.path.join(path_easy, 'hard_images_test.h5'))

# Ridimensiona le immagini facili
X_easy_train_resized = resize_images(X_easy_train)
X_easy_valid_resized = resize_images(X_easy_valid)
X_easy_test_resized = resize_images(X_easy_test)

# Ridimensiona le immagini difficili
X_hard_train_resized = resize_images(X_hard_train)
X_hard_valid_resized = resize_images(X_hard_valid)
X_hard_test_resized = resize_images(X_hard_test)

# Applica l'aumento dei dati alle immagini facili
datagen_easy = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen_easy.fit(X_easy_train_resized)

# Applica l'aumento dei dati alle immagini difficili
datagen_hard = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen_hard.fit(X_hard_train_resized)

# Definizione del modello semplice (ad esempio, MobileNetV2) con regolarizzazione e dropout
base_model_simple = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model_simple.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions_simple = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model_simple = tf.keras.Model(inputs=base_model_simple.input, outputs=predictions_simple)

# Definizione del modello avanzato (ad esempio, ResNet50) con regolarizzazione e dropout
base_model_advanced = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model_advanced.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions_advanced = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model_advanced = tf.keras.Model(inputs=base_model_advanced.input, outputs=predictions_advanced)

# Compila i modelli
model_simple.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_advanced.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Addestra il modello semplice con le immagini facili
history_simple = model_simple.fit(datagen_easy.flow(X_easy_train_resized, y_easy_train, batch_size=BATCH_SIZE), 
                                  validation_data=(X_easy_valid_resized, y_easy_valid), 
                                  epochs=TRAIN_EPOCH, 
                                  callbacks=[early_stopping])

# Addestra il modello avanzato con le immagini difficili
history_advanced = model_advanced.fit(datagen_hard.flow(X_hard_train_resized, y_hard_train, batch_size=BATCH_SIZE), 
                                      validation_data=(X_hard_valid_resized, y_hard_valid), 
                                      epochs=TRAIN_EPOCH, 
                                      callbacks=[early_stopping])

# Predici le etichette per le immagini facili
easy_test_pred = model_simple.predict(X_easy_test_resized)

# Predici le etichette per le immagini difficili
hard_test_pred = model_advanced.predict(X_hard_test_resized)

# Combina le predizioni e le etichette vere per il set di test
combined_test_pred = np.concatenate([np.argmax(easy_test_pred, axis=1), np.argmax(hard_test_pred, axis=1)])
combined_test_true = np.concatenate([y_easy_test, y_hard_test])

# Calcola l'accuratezza
test_accuracy = accuracy_score(combined_test_true, combined_test_pred)
print(f"Accuratezza sul set di test: {test_accuracy}")

# Calcola il report di classificazione
print("Report di classificazione sul set di test:")
print(classification_report(combined_test_true, combined_test_pred))

# Calcola e visualizza la matrice di confusione
conf_matrix = confusion_matrix(combined_test_true, combined_test_pred)
ConfusionMatrixDisplay(conf_matrix).plot()
plt.show()

# Fine-tuning del modello avanzato
print("Fine-tuning del modello avanzato...")
unfreeze = 59  # Sblocca più livelli per il fine-tuning
base_model_advanced.trainable = True
fine_tune_from = len(base_model_advanced.layers) - unfreeze
for layer in base_model_advanced.layers[:fine_tune_from]:
    layer.trainable = False
for layer in base_model_advanced.layers[fine_tune_from:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Compila nuovamente il modello con un tasso di apprendimento più basso
model_advanced.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FT_LR), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Continua l'addestramento del modello con il fine-tuning
history_finetune = model_advanced.fit(datagen_hard.flow(X_hard_train_resized, y_hard_train, batch_size=BATCH_SIZE), 
                                      validation_data=(X_hard_valid_resized, y_hard_valid), 
                                      epochs=FT_EPOCH, 
                                      callbacks=[early_stopping])

# Predici le etichette per i dati di test difficili
hard_test_pred = model_advanced.predict(X_hard_test_resized)

# Combina le predizioni dei modelli (ad esempio, media delle predizioni)
combined_test_pred = np.concatenate([np.argmax(easy_test_pred, axis=1), np.argmax(hard_test_pred, axis=1)])
combined_test_true = np.concatenate([y_easy_test, y_hard_test])

# Calcola l'accuratezza complessiva
test_accuracy = accuracy_score(combined_test_true, combined_test_pred)
print(f'Accuratezza sul set di test combinato: {test_accuracy}')

# Calcola il report di classificazione
print("Report di classificazione sul set di test combinato:")
print(classification_report(combined_test_true, combined_test_pred))

# Calcola e visualizza la matrice di confusione
conf_matrix = confusion_matrix(combined_test_true, combined_test_pred)
ConfusionMatrixDisplay(conf_matrix).plot()
plt.show()

# Salvataggio del modello
final_model_dir = os.path.join("final_models/preprocessing/SMOTE", dataset_name)
os.makedirs(final_model_dir, exist_ok=True)
model_name = os.path.join(final_model_dir, f"{dataset_name}_model_finetuned.keras")
model_advanced.save(model_name)
print(f"Modello salvato in: {model_name}")

# Ottieni le predizioni del modello sui dati di test
y_pred_prob = model_advanced.predict(X_hard_test_resized)
y_pred = np.argmax(y_pred_prob, axis=1)

# Confronta le predizioni con le etichette reali
correct_predictions = np.sum(y_pred == y_hard_test)
incorrect_predictions = np.sum(y_pred != y_hard_test)

# Stampa i risultati
print(f"Numero di predizioni corrette: {correct_predictions}")
print(f"Numero di predizioni sbagliate: {incorrect_predictions}")

# Calcola l'accuratezza manualmente per verifica
accuracy = correct_predictions / len(y_hard_test)
print(f"Accuratezza calcolata manualmente: {accuracy*100}%")

# Create directory for saving plots
results_dir = os.path.join("results/preprocessing/SMOTE", dataset_name)
os.makedirs(results_dir, exist_ok=True)

# Calcola la matrice di confusione
cm = confusion_matrix(y_hard_test, y_pred)

# Visualizza e salva la matrice di confusione con etichette    
classNames = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classNames)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

# Rotate the x-axis labels to avoid overlap
plt.xticks(rotation=25, ha='right')
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.close()

# Estrai le metriche di accuratezza e perdita
train_accuracy = history_finetune.history['accuracy']
val_accuracy = history_finetune.history['val_accuracy']
train_loss = history_finetune.history['loss']
val_loss = history_finetune.history['val_loss']

plt.figure(figsize=(12, 4))

# Grafico di Accuratezza
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Grafico di Perdita
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.savefig(os.path.join(results_dir, 'training_validation_plots.png'))
plt.show()

# Salvataggio dei parametri
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
    "accuracy test set": test_accuracy,
}

params_path = os.path.join(results_dir, 'parameters.txt')
with open(params_path, 'w') as f:
    for key, value in params.items():
        f.write(f"{key}: {value}\n")

print(f"Parametri salvati in: {params_path}")