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
# Usa la funzione schedule nel callback LearningRateScheduler
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from tensorflow.keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

############ modello senza batch normalization prima di drop in patch extraction 


# Carica i migliori iperparametri dal file
with open('best_hyperparameters.txt', 'r') as f:
    best_hps = {}
    for line in f:
        name, value = line.strip().split(': ')
        if name in ['units']:
            best_hps[name] = int(float(value))  # Converti esplicitamente a int
        else:
            best_hps[name] = float(value)

########################### MIO CODICE ########################################

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
FT_LR = best_hps['learning_rate']
FT_LR_DECAY_STEP = 80.0
FT_LR_DECAY_RATE = 1
FT_ES_PATIENCE = 20 #numero di epoche di tolleranza per l'arresto anticipato
FT_DROPOUT = best_hps['train_dropout']
dropout_rate = best_hps['dropout_rate']

ES_LR_MIN_DELTA = 0.003 #quantità minima di cambiamento per considerare un miglioramento

dataset_name='Bosphorus'

####################################################################################

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

# Supponiamo che i tuoi dati siano memorizzati in un file HDF5 chiamato 'data.h5'
with h5py.File(file_output, 'r') as f:
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

# Load your data here, PAtt-Lite was trained with h5py for shorter loading time
X_train, y_train = shuffle(X_train, y_train)

print("Shape of train_sample: {}".format(X_train.shape))
print("Shape of train_label: {}".format(y_train.shape))
print("Shape of valid_sample: {}".format(X_valid.shape))
print("Shape of valid_label: {}".format(y_valid.shape))
print("Shape of test_sample: {}".format(X_test.shape))
print("Shape of test_label: {}".format(y_test.shape))

# Il codice inizia calcolando i pesi delle classi per un dataset sbilanciato utilizzando la funzione compute_class_weight con l'opzione 'balanced'.
# I pesi delle classi vengono poi convertiti in un dizionario tramite dict(enumerate(class_weights)). 
# Questo è utile per bilanciare l'influenza delle classi durante l'addestramento del modello.

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Model Building

# Il modello inizia con un livello di input definito da tf.keras.Input con una forma specificata da IMG_SHAPE. 
# Viene applicato un ridimensionamento delle immagini a 224x224 pixel e un'augmentazione dei dati tramite un livello sequenziale che include 
# operazioni di flip orizzontale e 
# contrasto casuale. La funzione preprocess_input di MobileNet viene utilizzata per pre-processare i dati di input.

input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
sample_resizing = tf.keras.layers.Resizing(224, 224, name="resize")
data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode='horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(factor=0.3)
    ], name="augmentation")
preprocess_input = tf.keras.applications.mobilenet.preprocess_input

# Il backbone del modello è MobileNet, pre-addestrato su ImageNet, con i pesi congelati per evitare l'aggiornamento durante l'addestramento iniziale. 
# Viene creato un modello di base che estrae le caratteristiche fino a un certo livello di MobileNet. 
# Viene aggiunto un livello di attenzione per migliorare l'estrazione delle caratteristiche rilevanti,
# seguito da una serie di convoluzioni separabili per l'estrazione delle patch. 
# Un livello di pooling globale e un livello di pre-classificazione con una rete sequenziale vengono aggiunti prima del 
# livello di previsione finale che utilizza una funzione di attivazione softmax per la classificazione.

backbone = tf.keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
backbone.trainable = False
base_model = tf.keras.Model(backbone.input, backbone.layers[-29].output, name='base_model')

# Il modello viene compilato con l'ottimizzatore Adam e la funzione di perdita sparse_categorical_crossentropy. 
# Durante l'addestramento, vengono utilizzati callback per l'arresto anticipato e la riduzione del tasso di apprendimento
# in base alla precisione di validazione. Il modello viene addestrato sui dati di training e validazione, e successivamente valutato sui dati di test.

self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')
patch_extraction = tf.keras.Sequential([
    tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'), 
    tf.keras.layers.Dropout(dropout_rate),  # Aggiungi dropout
    tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'), 
    tf.keras.layers.Dropout(dropout_rate),  # Aggiungi dropout
    tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(best_hps['l2_reg']))
], name='patch_extraction')
global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
pre_classification = tf.keras.Sequential([
    tf.keras.layers.Dense(best_hps['units'], activation='relu', kernel_regularizer=l2(best_hps['l2_reg'])), 
    tf.keras.layers.BatchNormalization(),  
    tf.keras.layers.Dropout(dropout_rate)  # Aggiungi dropout
], name='pre_classification')

prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid", name='classification_head')

############################### MIO CODICE ########################################
inputs = input_layer
x = sample_resizing(inputs)
x = data_augmentation(x)
x = preprocess_input(x)
x = base_model(x, training=False)
x = patch_extraction(x)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(TRAIN_DROPOUT)(x)
x = pre_classification(x)
# Usa il nuovo livello nel tuo modello
x = ExpandDimsLayer(axis=1)(x)  # Aggiungi una dimensione di sequenza 
x = self_attention([x, x])
# Usa il nuovo livello nel tuo modello
x = SqueezeLayer(axis=1)(x)  # Rimuovi la dimensione di sequenza dopo l'attenzione
outputs = prediction_layer(x)

############################ VECCHIO CODICE ########################################

# inputs = input_layer
# x = sample_resizing(inputs)
# x = data_augmentation(x)
# x = preprocess_input(x)
# x = base_model(x, training=False)
# x = patch_extraction(x)
# x = global_average_layer(x)
# x = tf.keras.layers.Dropout(TRAIN_DROPOUT)(x)
# x = pre_classification(x)
# x = self_attention([x, x])
# outputs = prediction_layer(x)

####################################################################################

# Si sta utilizzando MobileNet come backbone e aggiungendo ulteriori livelli per adattarlo al task specifico 
# (classificazione delle espressioni facciali nel dataset CK+).
# Durante l'addestramento iniziale, si sta addestrando solo i nuovi livelli aggiunti, 
# mentre i pesi di MobileNet rimangono congelati (non vengono aggiornati).
# L'accuratezza che si ottiene dopo questo addestramento iniziale riflette le prestazioni del modello 
# sui dati di training e validation del dataset CK+.

model = tf.keras.Model(inputs, outputs, name='train-head')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=TRAIN_LR, global_clipnorm=3.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Training Procedure
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True)
learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)
history = model.fit(X_train, y_train, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_valid, y_valid), verbose=0, 
                    class_weight=class_weights, callbacks=[early_stopping_callback, learning_rate_callback])
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_acc}")

# Model Finetuning

# Dopo l'addestramento iniziale, il modello viene affinato. Viene sbloccata una parte del backbone di MobileNet per consentire l'addestramento 
# fine-tuning, mantenendo congelati i livelli di BatchNormalization. 
# Viene ricostruito il modello con dropout spaziale e regolare per prevenire l'overfitting. 
# Il modello viene nuovamente compilato e addestrato con callback aggiuntivi per il monitoraggio del tasso di apprendimento e il
# logging con TensorBoard. Infine, il modello affinato viene valutato sui dati di test e salvato su disco.

# Quindi, si sblocca una parte del backbone di MobileNet per consentire l'addestramento (fine-tuning).
# Durante il fine-tuning, i pesi di MobileNet vengono aggiornati insieme ai nuovi livelli aggiunti, 
# permettendo al modello di adattarsi meglio alle caratteristiche specifiche del dataset.

print("\nFinetuning ...")
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
def schedule(epoch, lr):
    if epoch < FT_LR_DECAY_STEP:
        return float(lr)
    else:
        return float(lr * FT_LR_DECAY_RATE)

# Definisci i callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=ES_LR_MIN_DELTA, patience=FT_ES_PATIENCE, restore_best_weights=True)
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule=schedule)

# Directory per salvare i pesi del modello
checkpoint_dir = os.path.join("checkpoints", dataset_name)
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.weights.h5")

checkpoint_callback = ModelCheckpoint(
    filepath='model_weights_epoch_{epoch:02d}.h5',  # Percorso del file di salvataggio
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
    X_train, y_train, 
    epochs=FT_EPOCH, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_valid, y_valid), 
    verbose=1, 
    initial_epoch=history.epoch[-TRAIN_ES_PATIENCE], 
    callbacks=[early_stopping_callback, scheduler_callback, tensorboard_callback, checkpoint_callback]
)

test_loss, test_acc = model.evaluate(X_test, y_test)


# Create directory for saving the final model
final_model_dir = os.path.join("final_models", dataset_name)
os.makedirs(final_model_dir, exist_ok=True)

# Save the model in the specified directory with .keras extension
model_name = os.path.join(final_model_dir, f"{dataset_name}_model.keras")
model.save(model_name)

# Save the weights of the model
weights_name = os.path.join(final_model_dir, f"{dataset_name}.weights.h5")
model.save_weights(weights_name)

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
results_dir = os.path.join("results", dataset_name)
os.makedirs(results_dir, exist_ok=True)

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

plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.close()

# Accedi alla storia dell'addestramento
history = history_finetune.history

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

plt.savefig(os.path.join(results_dir, 'training_validation_plots.png'))
plt.show()

# L'accuratezza che si ottiene prima del fine-tuning è quella del  modello addestrato sui dati del dataset analizzato, 
# utilizzando MobileNet come backbone pre-addestrato. 
# Il fine-tuning permette di migliorare ulteriormente le prestazioni del modello adattandolo meglio alle caratteristiche specifiche del  dataset.
