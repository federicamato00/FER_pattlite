import numpy as np
import h5py
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



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
FT_LR_DECAY_RATE = 0.5 #era a 1 ma ho messo 0.5
FT_DROPOUT = best_hps_ft['ft_dropout']  # Define FT_DROPOUT
dropout_rate = FT_DROPOUT  # Define dropout_rate
FT_ES_PATIENCE = 20 #numero di epoche di tolleranza per l'arresto anticipato
FT_DROPOUT = best_hps['train_dropout']
dropout_rate = best_hps['dropout_rate']

ES_LR_MIN_DELTA = 0.003 #quantità minima di cambiamento per considerare un miglioramento

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
predictions_simple = tf.keras.layers.Dense(7, activation='softmax')(x)
model_simple = tf.keras.Model(inputs=base_model_simple.input, outputs=predictions_simple)

# Definizione del modello avanzato (ad esempio, ResNet50) con regolarizzazione e dropout
base_model_advanced = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model_advanced.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions_advanced = tf.keras.layers.Dense(7, activation='softmax')(x)
model_advanced = tf.keras.Model(inputs=base_model_advanced.input, outputs=predictions_advanced)

# Compila i modelli
model_simple.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_advanced.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Addestra il modello semplice con le immagini facili
history_simple = model_simple.fit(datagen_easy.flow(X_easy_train_resized, y_easy_train, batch_size=BATCH_SIZE), validation_data=(X_easy_valid_resized, y_easy_valid), epochs=TRAIN_EPOCH, callbacks=[early_stopping])

# Addestra il modello avanzato con le immagini difficili
history_advance = model_advanced.fit(datagen_hard.flow(X_hard_train_resized, y_hard_train, batch_size=BATCH_SIZE), validation_data=(X_hard_valid_resized, y_hard_valid), epochs=TRAIN_EPOCH, callbacks=[early_stopping])




# Predici le etichette per le immagini facili
easy_train_pred = model_simple.predict(X_easy_train_resized)
easy_valid_pred = model_simple.predict(X_easy_valid_resized)
easy_test_pred = model_simple.predict(X_easy_test_resized)

# Predici le etichette per le immagini difficili
hard_train_pred = model_advanced.predict(X_hard_train_resized)
hard_valid_pred = model_advanced.predict(X_hard_valid_resized)
hard_test_pred = model_advanced.predict(X_hard_test_resized)

# Combina le predizioni e le etichette vere per il set di addestramento
combined_train_pred = np.concatenate([np.argmax(easy_train_pred, axis=1), np.argmax(hard_train_pred, axis=1)])
combined_train_true = np.concatenate([y_easy_train, y_hard_train])

# Combina le predizioni e le etichette vere per il set di validazione
combined_valid_pred = np.concatenate([np.argmax(easy_valid_pred, axis=1), np.argmax(hard_valid_pred, axis=1)])
combined_valid_true = np.concatenate([y_easy_valid, y_hard_valid])

# Combina le predizioni e le etichette vere per il set di test
combined_test_pred = np.concatenate([np.argmax(easy_test_pred, axis=1), np.argmax(hard_test_pred, axis=1)])
combined_test_true = np.concatenate([y_easy_test, y_hard_test])


combined_train = np.concatenate([X_easy_train_resized, X_hard_train_resized])
combined_valid = np.concatenate([X_easy_valid_resized, X_hard_valid_resized])
combined_test = np.concatenate([X_easy_test_resized, X_hard_test_resized])
combined_label_train = np.concatenate([y_easy_train, y_hard_train])
combined_label_valid = np.concatenate([y_easy_valid, y_hard_valid])
combined_label_test = np.concatenate([y_easy_test, y_hard_test])

# Calcola l'accuratezza
train_accuracy = accuracy_score(combined_train_true, combined_train_pred)
valid_accuracy = accuracy_score(combined_valid_true, combined_valid_pred)
test_accuracy = accuracy_score(combined_test_true, combined_test_pred)

print(f"Accuratezza sul set di addestramento: {train_accuracy}")
print(f"Accuratezza sul set di validazione: {valid_accuracy}")
print(f"Accuratezza sul set di test: {test_accuracy}")

# Calcola il report di classificazione
print("Report di classificazione sul set di test:")
print(classification_report(combined_test_true, combined_test_pred))

# Calcola e visualizza la matrice di confusione
conf_matrix = confusion_matrix(combined_test_true, combined_test_pred)
ConfusionMatrixDisplay(conf_matrix).plot()
plt.show()


print("Finetuning...")
backbone = tf.keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
backbone.trainable = False
base_model = tf.keras.Model(backbone.input, backbone.layers[-29].output, name='base_model')
sample_resizing = tf.keras.layers.Resizing(224, 224, name="resize")
data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode='horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(factor=0.3)
    ], name="augmentation")
    
preprocess_input = tf.keras.applications.mobilenet.preprocess_input
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


self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')
prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name='classification_head')
unfreeze = 59  # Sbloccato più livelli per il fine-tuning
base_model.trainable = True
fine_tune_from = len(base_model.layers) - unfreeze
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_from:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

################################## MIO CODICE ########################################
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
x = ExpandDimsLayer(axis=1)(x)  # Aggiungi una dimensione di sequenza
x = self_attention([x, x])
x = SqueezeLayer(axis=1)(x)  # Rimuovi la dimensione di sequenza dopo l'attenzione
x = tf.keras.layers.Dropout(FT_DROPOUT)(x)

outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs, name='finetune-backbone')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FT_LR, global_clipnorm=3.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Training Procedure

# Definisci la funzione di schedule
# Definisci la funzione di Cosine Annealing
def schedule(epoch, FT_LR):
    return 0.5 * (1 + np.cos(np.pi * epoch / FT_EPOCH)) * FT_LR


# Definisci i callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=ES_LR_MIN_DELTA, patience=FT_ES_PATIENCE, restore_best_weights=True)
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule=schedule)

# Directory per salvare i pesi del modello
checkpoint_dir = os.path.join("checkpoints/preprocessing/SMOTE", dataset_name)


# Callback per salvare i pesi del modello ogni 20 epoche

checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir,'model_weights_epoch_{epoch:02d}.weights.h5'),  # Percorso del file di salvataggio
    save_weights_only=True,  # Salva solo i pesi del modello
    save_best_only=True,  # Salva solo il miglior modello

)
# Carica i pesi del modello dal checkpoint più recente
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    model.load_weights(latest_checkpoint)
    print(f"Pesi del modello caricati da: {latest_checkpoint}")

# Continua l'addestramento
history_finetune = model.fit(
    combined_train, combined_label_train, 
    epochs=FT_EPOCH, 
    batch_size=BATCH_SIZE, 
    validation_data=(combined_valid, combined_label_valid), 
    verbose=1, 
    initial_epoch=history.epoch[-TRAIN_ES_PATIENCE], 
    callbacks=[early_stopping_callback, scheduler_callback, tensorboard_callback, checkpoint_callback]
)

test_loss, test_acc = model.evaluate(X_test, y_test)

# Function to create a unique directory
def create_unique_directory(base_dir):
    import uuid
    unique_id = str(uuid.uuid4())
    unique_dir = os.path.join(base_dir, unique_id)
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir

# Create directory for saving the final model
final_model_dir = os.path.join("final_models/preprocessing/SMOTE", dataset_name)

# Creazione della directory unica per i risultati
base_dir = final_model_dir
unique_dir = create_unique_directory(base_dir)


# Save the model in the specified directory with .keras extension
model_name = os.path.join(unique_dir, f"{dataset_name}_model.keras")
model.save(model_name)

print(f"Modello salvato in: {model_name}")

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Ottieni le predizioni del modello sui dati di test
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Confronta le predizioni con le etichette reali
correct_predictions = np.sum(y_pred == y_test)
incorrect_predictions = np.sum(y_pred != y_test)

# Stampa i risultati
print(f"Numero di predizioni corrette: {correct_predictions}")
print(f"Numero di predizioni sbagliate: {incorrect_predictions}")

# Calcola l'accuratezza manualmente per verifica
accuracy = correct_predictions / len(y_test)
print(f"Accuratezza calcolata manualmente: {accuracy*100}%")

# Create directory for saving plots
results_dir = os.path.join("results/preprocessing/SMOTE", dataset_name)

# Calcola la matrice di confusione
cm = confusion_matrix(y_test, y_pred)

# Visualizza e salva la matrice di confusione con etichette    
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

# Rotate the x-axis labels to avoid overlap
plt.xticks(rotation=25, ha='right')
# Creazione della directory unica per i risultati
base_dir = results_dir
unique_dir = create_unique_directory(base_dir)


plt.savefig(os.path.join(unique_dir, 'confusion_matrix.png'))
plt.close()



# Estrai le metriche di accuratezza e perdita
train_accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
train_loss = history['loss']
val_loss = history['val_loss']

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


plt.savefig(os.path.join(unique_dir, 'training_validation_plots.png'))
plt.show()

# L'accuratezza che si ottiene prima del fine-tuning è quella del  modello addestrato sui dati del dataset analizzato, 
# utilizzando MobileNet come backbone pre-addestrato. 
# Il fine-tuning permette di migliorare ulteriormente le prestazioni del modello adattandolo meglio alle caratteristiche specifiche del  dataset.



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
    "pre_classification": pre_classification.get_config(),
    "patch_extraction": patch_extraction.get_config(),
    "accuracy test set": test_acc,
    "accuracy train set": train_accuracy[-1],
    "accuracy validation set": val_accuracy[-1],
}

save_parameters(params, unique_dir)
print(f"Directory creata: {unique_dir}")
print(f"Parametri salvati in: {os.path.join(unique_dir, 'parameters.txt')}")