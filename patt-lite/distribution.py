import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

# Funzione per caricare le immagini e le etichette
def load_images_and_labels(file_path, save_path):
    with h5py.File(file_path, 'r') as f:
        if save_path in ['processed_bosphorus_5.h5', 'processed_ckplus.h5', 'ckplus_baseD_processed.h5']:
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

dataset_name = 'CK+'
# Visualizza e salva la matrice di confusione con etichette
if 'RAFDB' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
elif 'FERP' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
elif 'JAFFE' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
elif 'Bosphorus' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
elif 'CK+' in dataset_name:
    classNames = ['neutral', 'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']  # 7 classi
else:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

seven_classes = dataset_name + '_numClasses7'
dataset_path = os.path.join('datasets', 'processed', seven_classes, 'processed_ckplus.h5')
X_train, y_train, X_valid, y_valid, X_test, y_test = load_images_and_labels(dataset_path, 'processed_ckplus.h5')

# Funzione per visualizzare la distribuzione delle etichette
def plot_label_distribution(y, classNames, title):
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(classNames, counts)
    plt.xlabel('Classi')
    plt.ylabel('Numero di campioni')
    plt.title(title)
    plt.show()

# Visualizza la distribuzione delle etichette per ogni set di dati
plot_label_distribution(y_train, classNames, 'Distribuzione delle etichette nel set di addestramento')
plot_label_distribution(y_valid, classNames, 'Distribuzione delle etichette nel set di validazione')
plot_label_distribution(y_test, classNames, 'Distribuzione delle etichette nel set di test')