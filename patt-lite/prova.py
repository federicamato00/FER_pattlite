
import datetime
import keras
import numpy as np
import h5py
from sklearn.utils import compute_class_weight, shuffle
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Layer


def create_unique_directory(base_dir):
    """
    Crea una directory unica aggiungendo un numero di riferimento se la directory esiste già.
    """
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
    """
    Salva i parametri in un file .txt nella directory specificata.
    """
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")


############ modello con batch normalization prima di drop in patch extraction 

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
FT_LR = best_hps_ft['ft_learning_rate']
FT_LR_DECAY_STEP = 80.0
FT_LR_DECAY_RATE = 0.5 #era a 1 ma ho messo 0.5

######
# Epochs successive: Se l'epoca corrente è maggiore o uguale a FT_LR_DECAY_STEP,
# la funzione riduce il learning rate moltiplicandolo per 0.5. Questo significa che il 
# learning rate viene dimezzato per rendere l'addestramento più fine e stabile nelle fasi successive.

FT_ES_PATIENCE = 20 #numero di epoche di tolleranza per l'arresto anticipato
FT_DROPOUT = best_hps['train_dropout']
dropout_rate = best_hps['dropout_rate']

ES_LR_MIN_DELTA = 0.003 #quantità minima di cambiamento per considerare un miglioramento

dataset_name='Bosphorus'



# Funzione per caricare le immagini e le etichette
def load_images_and_labels(file_path):
    with h5py.File(file_path, 'r') as f:
        X_train = np.array(f['X_train'])
        y_train = np.array(f['y_train'])
        if file_path=='modified_dataset.h5':
            X_valid = np.array(f['X_valid'])
            y_valid = np.array(f['y_valid'])
        else:
            X_valid = np.array(f['X_val'])
            y_valid = np.array(f['y_val'])
        X_test = np.array(f['X_test'])
        y_test = np.array(f['y_test'])
    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Funzione per ridimensionare le immagini
def resize_images(X, target_size=(120, 120)):
    return np.array([tf.image.resize(image, target_size).numpy() for image in X])

# Carica le immagini 
path_easy = 'datasets/preprocessing/Bosphorus/SMOTE'

X_train, y_train , X_valid, y_valid, X_test, y_test= load_images_and_labels( 'modified_dataset.h5')


# Ridimensiona le immagini facili
X_train = resize_images(   X_train)
X_valid = resize_images(X_valid)
X_test = resize_images(X_test)

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


# # Carica le immagini preprocessate
# def load_preprocessed_data(file_path):
#     with h5py.File(file_path, 'r') as f:
#         X = np.array(f['X'])
#         y = np.array(f['y'])
#     return X, y

# Percorsi ai file preprocessati
dataset_file_path = os.path.join('datasets', dataset_name, 'bosphorus_SMOTE.h5')
# valid_file_path = os.path.join('datasets', 'preprocessing', dataset_name, 'SMOTE', 'valid.h5')
# test_file_path = os.path.join('datasets', 'preprocessing', dataset_name, 'SMOTE', 'test.h5')

# # Carica i dati preprocessati
X_train_original, y_train_original, X_valid_original, y_valid_original, X_test_original, y_test_original = load_images_and_labels(dataset_file_path)
# X_valid_preprocessed, y_valid_preprocessed = load_preprocessed_data(valid_file_path)
# X_test_preprocessed, y_test_preprocessed = load_preprocessed_data(test_file_path)

# Verifica visiva delle immagini preprocessate
def visualize_images(original_images,preprocessed_images, num_images=5):
    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))
    for i in range(num_images):
        axes[0, i].imshow(original_images[i])
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(preprocessed_images[i])
        axes[1, i].set_title('Preprocessed')
        axes[1, i].axis('off')
    plt.show()

# Esempio di utilizzo
visualize_images(X_train_original[:5],X_train[:5])

# Load your data here, PAtt-Lite was trained with h5py for shorter loading time
X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)
X_test, y_test = shuffle(X_test, y_test)    


class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Model Building

# Il modello inizia con un livello di input definito da tf.keras.Input con una forma specificata da IMG_SHAPE. 
# Viene applicato un ridimensionamento delle immagini a 224x224 pixel e un'augmentazione dei dati tramite un livello sequenziale che include 
# operazioni di flip orizzontale e 
# contrasto casuale. La funzione preprocess_input di MobileNet viene utilizzata per pre-processare i dati di input.

input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
# Ridimensiona le immagini a (120, 120, 3)
X_train = resize_images(X_train, target_size=(120, 120))
X_valid = resize_images(X_valid, target_size=(120, 120))
X_test = resize_images(X_test, target_size=(120, 120))

print("Shape of train_sample: {}".format(X_train.shape))
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

####### con dropout e batch normalization prima di drop in patch extraction ########
#accuracy =  su test set

# patch_extraction = tf.keras.Sequential([
#     tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'), 
#     tf.keras.layers.BatchNormalization(),  
#     tf.keras.layers.Dropout(dropout_rate),  # Aggiungi dropout
#     tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'), 
#     tf.keras.layers.BatchNormalization(),  
#     tf.keras.layers.Dropout(dropout_rate),  # Aggiungi dropout
#     tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(best_hps['l2_reg']))
# ], name='patch_extraction')

########################## modello iniziale ######################################## 
# accuracy = 80.33% su test set

patch_extraction = tf.keras.Sequential([
    
    tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'), 

    tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'), 

    tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(best_hps['l2_reg']))
], name='patch_extraction')


########################## modello con dropout ma senza batch ######################

# accuracy = 79% su test set
# patch_extraction = tf.keras.Sequential([
#     tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'), 
#     tf.keras.layers.Dropout(dropout_rate),  # Aggiungi dropout
#     tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'), 
#     tf.keras.layers.Dropout(dropout_rate),  # Aggiungi dropout
#     tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(best_hps['l2_reg']))
# ], name='patch_extraction')


####################################################################################

global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
pre_classification = tf.keras.Sequential([
    tf.keras.layers.Dense(best_hps['units'], activation='relu', kernel_regularizer=l2(best_hps['l2_reg'])), 
    tf.keras.layers.BatchNormalization(),  
    tf.keras.layers.Dropout(dropout_rate)  # Aggiungi dropout
], name='pre_classification')

prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name='classification_head')


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
# Definisci la funzione di Cosine Annealing
def schedule(epoch, lr):
    return 0.5 * (1 + np.cos(np.pi * epoch / FT_EPOCH)) * FT_LR


# Definisci i callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=ES_LR_MIN_DELTA, patience=FT_ES_PATIENCE, restore_best_weights=True)
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule=schedule)

# Directory per salvare i pesi del modello
checkpoint_dir = os.path.join("checkpoints/PROVA_2", dataset_name)


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
final_model_dir = os.path.join("final_models/PROVA_2", dataset_name)

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
results_dir = os.path.join("results/PROVA_2", dataset_name)

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