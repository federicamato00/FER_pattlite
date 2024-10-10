import numpy as np
import matplotlib.pyplot as plt
import h5py

def plot_class_distribution(y_train, y_val, y_test, class_names):
    # Conta il numero di campioni per ciascuna classe in ogni set
    train_counts = np.bincount(y_train)
    val_counts = np.bincount(y_val)
    test_counts = np.bincount(y_test)

    # Crea un array con gli indici delle classi
    classes = np.arange(len(class_names))

    # Imposta la larghezza delle barre
    bar_width = 0.25

    # Crea la figura e gli assi
    fig, ax = plt.subplots(figsize=(12, 6))

    # Crea le barre per il set di training
    ax.bar(classes - bar_width, train_counts, width=bar_width, label='Train')

    # Crea le barre per il set di validazione
    ax.bar(classes, val_counts, width=bar_width, label='Validation')

    # Crea le barre per il set di test
    ax.bar(classes + bar_width, test_counts, width=bar_width, label='Test')

    # Aggiungi le etichette e il titolo
    ax.set_xlabel('Classi')
    ax.set_ylabel('Numero di campioni')
    ax.set_title('Distribuzione delle classi nei set di training, validazione e test')
    ax.set_xticks(classes)
    ax.set_xticklabels(class_names)
    ax.legend()
    plt.savefig('class_distribution_per_set.png')
    # Mostra il grafico
    plt.show()


# Funzione per caricare le immagini e le etichette
def load_images_and_labels(file_path):
    with h5py.File(file_path, 'r') as f:
        
        if file_path=='processed_bosphorus_5.h5':
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


# Carica le immagini 
path_easy = 'datasets/preprocessing/Bosphorus/SMOTE'

X_train, y_train , X_valid, y_valid, X_test, y_test= load_images_and_labels( 'bosphorus_data_augmentation_5.h5')

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

# Esempio di utilizzo
class_names = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
plot_class_distribution(y_train, y_valid, y_test, class_names)