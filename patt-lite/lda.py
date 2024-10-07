import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def create_scatter_plot(X, y, classNames, title, path, use_lda=False):
    if use_lda:
        lda = LDA(n_components=2)
        X_transformed = lda.fit_transform(X, y)
    else:
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X.reshape((X.shape[0], -1)))
    
    plt.figure(figsize=(10, 5))
    for class_idx in np.unique(y):
        plt.scatter(X_transformed[y == class_idx, 0], X_transformed[y == class_idx, 1], label=classNames[class_idx], alpha=0.5)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.savefig(path)
    plt.close()

def apply_lda(X, y):
    lda = LDA(n_components=6)
    X_lda = lda.fit_transform(X.reshape((X.shape[0], -1)), y)
    print(f"X_lda shape: {X_lda.shape}")
    return X_lda

def load_data(
    path_prefix,
    dataset_name,
    test_size=0.2,
    val_size=0.1,
):
    X, y = [], []

    IMG_SIZE = 224 if 'RAFDB' in dataset_name else 120

    if 'RAFDB' in dataset_name:
        classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    elif 'FERP' in dataset_name:
        classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    elif 'JAFFE' in dataset_name:
        classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    elif 'Bosphorus' in dataset_name:
        classNames = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise','neutral']
    elif 'CK+' in dataset_name:
        classNames = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        
    else:
        classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

    PATH = os.path.join(path_prefix, dataset_name)
    path_distribution = os.path.join('dataset_distribution','prova', dataset_name)
    if dataset_name == 'CK+':
        for classes in os.listdir(PATH):
            if classes != '.DS_Store':
                class_path = os.path.join(PATH, classes)
                class_numeric = classNames.index(classes)

                for sample in os.listdir(class_path):
                    sample_path = os.path.join(class_path, sample)
                    image = cv2.imread(sample_path, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                    X.append(image)
                    y.append(class_numeric)
    elif dataset_name == 'Bosphorus':
        PATH = os.path.join(path_prefix, dataset_name, 'Bosphorus')
        for subject_folder in os.listdir(PATH):
            subject_path = os.path.join(PATH, subject_folder)
            if os.path.isdir(subject_path) and subject_folder != '.DS_Store':
                for file in os.listdir(subject_path):
                    if file.endswith('.png'):
                        parts = file.split('_')
                        if len(parts) > 2:
                            if parts[1] == 'E':  # Filtra solo le espressioni emotive
                                expression = parts[2].lower()
                            elif parts[1] == 'N':  # Gestisce l'espressione neutra
                                expression = 'neutral'
                            else:
                                continue  # Salta i file che non corrispondono ai criteri
    
                            if expression in [e.lower() for e in classNames]:  # Confronta con i nomi normalizzati
                                class_numeric = [e.lower() for e in classNames].index(expression)
                                file_path = os.path.join(subject_path, file)
                                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                                X.append(image)
                                y.append(class_numeric)
    else:
        raise ValueError("Tipo di dataset non supportato: {}".format(dataset_name))

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Visualizza la distribuzione delle classi prima di SMOTE
    plt.figure(figsize=(10, 5))
    plt.bar(np.unique(y), np.bincount(y))
    plt.title('Distribuzione delle classi prima di SMOTE')
    plt.xlabel('Classi')
    plt.ylabel('Numero di campioni')
    plt.xticks(ticks=np.unique(y), labels=[classNames[i] for i in np.unique(y)])
    plt.savefig(os.path.join(path_distribution, 'prima_SMOTE.png'))
    
     # Scatter plot prima di SMOTE
    create_scatter_plot(X, y, classNames, 'Distribuzione dei dati prima di SMOTE', os.path.join(path_distribution, 'scatter_prima_SMOTE.png'))
    
    # Controlla se le classi sono sbilanciate
    class_counts = np.bincount(y)
    max_count = np.max(class_counts)
    imbalance_threshold = 0.5  # Soglia per considerare una classe sbilanciata (10%)
    is_imbalanced = any(count < max_count * (1 - imbalance_threshold) for count in class_counts)
    
    if is_imbalanced:
        print("Le classi sono sbilanciate. Applicazione di SMOTE...")
        # Reshape X for SMOTE
        X_reshaped = X.reshape((X.shape[0], -1))
        
        # Apply SMOTE to the data
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
        
        # Reshape X back to original shape
        X_resampled = X_resampled.reshape((-1, IMG_SIZE, IMG_SIZE, 3))
        
        # Visualizza la distribuzione delle classi dopo SMOTE
        plt.figure(figsize=(10, 5))
        plt.bar(np.unique(y_resampled), np.bincount(y_resampled))
        plt.title('Distribuzione delle classi dopo SMOTE')
        plt.xlabel('Classi')
        plt.ylabel('Numero di campioni')
        plt.xticks(ticks=np.unique(y_resampled), labels=[classNames[i] for i in np.unique(y_resampled)])
        plt.savefig(os.path.join(path_distribution,'dopo_SMOTE.png'))
        
        # Scatter plot dopo SMOTE
        create_scatter_plot(X_resampled, y_resampled, classNames, 'Distribuzione dei dati dopo SMOTE', os.path.join(path_distribution, 'scatter_dopo_SMOTE.png'))
        
        X, y = X_resampled, y_resampled
    else:
        print("Le classi sono bilanciate. SMOTE non è necessario.")
    
    # Apply LDA
    X_lda = apply_lda(X, y)
    
    
    # Visualizza la distribuzione delle classi dopo LDA
    plt.figure(figsize=(10, 5))
    plt.bar(np.unique(y), np.bincount(y))
    plt.title('Distribuzione delle classi dopo LDA')
    plt.xlabel('Classi')
    plt.ylabel('Numero di campioni')
    plt.xticks(ticks=np.unique(y), labels=[classNames[i] for i in np.unique(y)])
    plt.savefig(os.path.join(path_distribution, 'dopo_LDA.png'))
    
    # Scatter plot dopo LDA
    create_scatter_plot(X_lda, y, classNames, 'Distribuzione dei dati dopo LDA', os.path.join(path_distribution, 'scatter_dopo_LDA.png'), use_lda=True)
    
    # Split the data into train, val, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_lda, y, test_size=test_size + val_size, random_state=42, stratify=y)
    val_size_adjusted = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp)
    
    return {'train': X_train, 'val': X_val, 'test': X_test}, {'train': y_train, 'val': y_val, 'test': y_test}


dataset_name='Bosphorus' # 'CK+', 'RAFDB', 'FERP', 'JAFFE', 'Bosphorus'
X, y = load_data('', dataset_name)
if 'CK+' in dataset_name:
    file_output = 'ckplus' 
elif 'RAFDB' in dataset_name:
    file_output = 'rafdb' 
elif 'FERP' in dataset_name:
    file_output = 'ferp'
elif 'JAFFE' in dataset_name:
    file_output = 'jaffe'
elif 'Bosphorus' in dataset_name:
    file_output = 'bosphorus'
else:
    file_output = 'dataset'

file_output = file_output + '_prova.h5'


with h5py.File(file_output, 'w') as dataset: 
    for split in X.keys():
        dataset.create_dataset(f'X_{split}', data=X[split])
        dataset.create_dataset(f'y_{split}', data=y[split])

del X, y