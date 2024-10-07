import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



def create_pairwise_scatter_plot_LDA(X, y, classNames, title, path, n_components=2):
    lda = LDA(n_components=n_components)
    X_transformed = lda.fit_transform(X, y)
    
    fig, axes = plt.subplots(n_components, n_components, figsize=(15, 15))
    fig.suptitle(title)
    
    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                ax = axes[i, j]
                for class_idx in np.unique(y):
                    ax.scatter(X_transformed[y == class_idx, i], X_transformed[y == class_idx, j], label=classNames[class_idx], alpha=0.5)
                if i == n_components - 1:
                    ax.set_xlabel(f'Component {j + 1}')
                if j == 0:
                    ax.set_ylabel(f'Component {i + 1}')
            else:
                axes[i, j].axis('off')
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.savefig(path)
    plt.close()


    
def create_scatter_plot_PCA(X, y, classNames, title, path, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X.reshape((X.shape[0], -1)))
    
     
    fig, axes = plt.subplots(n_components, n_components, figsize=(15, 15))
    fig.suptitle(title)
    
    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                ax = axes[i, j]
                for class_idx in np.unique(y):
                    ax.scatter(X_pca[y == class_idx, i], X_pca[y == class_idx, j], label=classNames[class_idx], alpha=0.5)
                if i == n_components - 1:
                    ax.set_xlabel(f'PCA Component {j + 1}')
                if j == 0:
                    ax.set_ylabel(f'PCA Component {i + 1}')
            else:
                axes[i, j].axis('off')
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.savefig(path)
    plt.close()

def choose_n_components_lda(X, y, variance_threshold=0.95):
    X = X.reshape((X.shape[0], -1))
    lda = LDA().fit(X, y)
    explained_variance_ratio = lda.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Trova il numero di componenti che spiegano almeno il variance_threshold della varianza
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Grafico della varianza spiegata cumulativa
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_variance, marker='o')
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.xlabel('Numero di Componenti')
    plt.ylabel('Varianza Spiegata Cumulativa')
    plt.title('Scelta del Numero di Componenti LDA')
    plt.savefig(os.path.join('dataset_distribution/Bosphorus', 'varianza_spiegata_cumulativa_LDA.png'))
    plt.close()
    return n_components

def choose_n_components_PCA(X, variance_threshold=0.95):
    pca = PCA().fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Trova il numero di componenti che spiegano almeno il variance_threshold della varianza
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Grafico della varianza spiegata cumulativa
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_variance, marker='o')
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.xlabel('Numero di Componenti')
    plt.ylabel('Varianza Spiegata Cumulativa')
    plt.title('Scelta del Numero di Componenti PCA')
    plt.savefig(os.path.join('dataset_distribution/Bosphorus', 'varianza_spiegata_cumulativa_PCA.png'))
    plt.close()
    
    return n_components

def apply_lda(X, y, n_components_lda):
    lda = LDA(n_components=n_components_lda)
    X_lda = lda.fit_transform(X.reshape((X.shape[0], -1)), y)
    return X_lda

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_unique_directory(base_path, name):
    path = os.path.join(base_path, name)
    counter = 1
    while os.path.exists(path):
        path = os.path.join(base_path, f"{name}_{counter}")
        counter += 1
    return path

def save_dataset(X, y, path, filename):
    create_directory(path)
    file_output = os.path.join(path, filename)
    with h5py.File(file_output, 'w') as dataset: 
        dataset.create_dataset('X', data=X)
        dataset.create_dataset('y', data=y)

def load_data(
    path_prefix,
    dataset_name,
    test_size=0.2,
    val_size=0.1,
    use_pca=False, 
    use_lda=False,
    use_smote=False,
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
    path_distribution = os.path.join('dataset_distribution/Bosphorus', dataset_name)
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
    
    # Creare le directory per salvare i dati e i grafici
    dataset_dir = get_unique_directory('datasets', dataset_name)
    create_directory(dataset_dir)
    create_directory(path_distribution)
    
   
    
    # Visualizza la distribuzione delle classi prima di SMOTE
    plt.figure(figsize=(10, 5))
    plt.bar(np.unique(y), np.bincount(y))
    plt.title('Distribuzione delle classi')
    plt.xlabel('Classi')
    plt.ylabel('Numero di campioni')
    plt.xticks(ticks=np.unique(y), labels=[classNames[i] for i in np.unique(y)])
    plt.savefig(os.path.join(path_distribution, 'distribution_dataset.png'))
    
    if use_smote:
        # Controlla se le classi sono sbilanciate
        class_counts = np.bincount(y)
        max_count = np.max(class_counts)
        imbalance_threshold = 0.5  # Soglia per considerare una classe sbilanciata (50%)
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
            
            X, y = X_resampled, y_resampled
        else:
            print("Le classi sono bilanciate. SMOTE non è necessario.")

    if use_pca:

    
        # # Scegliere il numero di componenti per PCA
        n_components = choose_n_components_PCA(X.reshape((X.shape[0], -1)))
        print(f"Numero di componenti PCA scelto: {n_components}")
        
        # # Applicare PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X.reshape((X.shape[0], -1)))
        if use_lda and not use_smote:
            title = 'PCA_scatter_PCA_LDA.png'
        if use_lda and use_smote:
            title = 'PCA_scatter_PCA_LDA_SMOTE.png'
        if not use_lda and not use_smote:
            title = 'PCA_scatter_PCA.png'
        if not use_lda and use_smote:
            title = 'PCA_scatter_PCA_SMOTE.png'
        create_scatter_plot_PCA(X, y, classNames, 'Distribuzione dei dati dopo PCA', os.path.join(path_distribution, title), n_components)

        X, y = X_pca, y
    
    if use_lda:

        # Apply LDA
        n_components_lda = choose_n_components_lda(X, y)
        X_lda = apply_lda(X, y, n_components_lda=n_components_lda)

        print(f"Numero di componenti LDA scelto: {n_components_lda}")
        
        # Visualizza la distribuzione delle classi dopo LDA
        plt.figure(figsize=(10, 5))
        plt.bar(np.unique(y), np.bincount(y))
        plt.title('Distribuzione delle classi dopo LDA')
        plt.xlabel('Classi')
        plt.ylabel('Numero di campioni')
        plt.xticks(ticks=np.unique(y), labels=[classNames[i] for i in np.unique(y)])
        plt.savefig(os.path.join(path_distribution, 'dopo_LDA.png'))
        if use_pca and not use_smote:
            title = 'LDA_scatter_PCA_LDA.png'
        if use_pca and use_smote:
            title = 'LDA_scatter_PCA_LDA_SMOTE.png'
        if not use_pca and not use_smote:
            title = 'LDA_scatter_LDA.png'
        if not use_pca and use_smote:
            title = 'LDA_scatter_LDA_SMOTE.png'
        

        # Scatter plot dopo LDA
        create_pairwise_scatter_plot_LDA(X_lda, y, classNames, 'Distribuzione dei dati dopo LDA', os.path.join(path_distribution, title), n_components_lda)
        X, y = X_lda, y

    # Split the data into train, val, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=42, stratify=y)
    val_size_adjusted = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp)
    
    return {'train': X_train, 'val': X_val, 'test': X_test}, {'train': y_train, 'val': y_val, 'test': y_test}


dataset_name='Bosphorus' # 'CK+', 'RAFDB', 'FERP', 'JAFFE', 'Bosphorus'


print("Loading data...")
X, y = load_data('', dataset_name, use_pca=False, use_lda=False, use_smote=False)
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

file_output = file_output + '.h5'

with h5py.File(file_output, 'w') as dataset: 
    for split in X.keys():
        dataset.create_dataset(f'X_{split}', data=X[split])
        dataset.create_dataset(f'y_{split}', data=y[split])

del X, y

print("Loading data with SMOTE...")

X, y = load_data('', dataset_name, use_pca=False, use_lda=False, use_smote=True)
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

file_output = file_output + '_SMOTE.h5'

save_path = os.path.join('datasets', dataset_name, file_output)

with h5py.File(save_path, 'w') as dataset: 
    for split in X.keys():
        dataset.create_dataset(f'X_{split}', data=X[split])
        dataset.create_dataset(f'y_{split}', data=y[split])

del X, y


print("Loading data with LDA...")


X, y = load_data('', dataset_name, use_pca=False, use_lda=True, use_smote=False)
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

file_output = file_output + '_LDA.h5'

save_path = os.path.join('datasets', dataset_name, file_output)

with h5py.File(save_path, 'w') as dataset: 
    for split in X.keys():
        dataset.create_dataset(f'X_{split}', data=X[split])
        dataset.create_dataset(f'y_{split}', data=y[split])

del X, y


print("Loading data with LDA and SMOTE...")

X, y = load_data('', dataset_name, use_pca=False, use_lda=True, use_smote=True)
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

file_output = file_output + '_LDA_SMOTE.h5'

save_path = os.path.join('datasets', dataset_name, file_output)

with h5py.File(save_path, 'w') as dataset: 
    for split in X.keys():
        dataset.create_dataset(f'X_{split}', data=X[split])
        dataset.create_dataset(f'y_{split}', data=y[split])

del X, y


# print("Loading data with PCA, LDA and SMOTE...")

# X, y = load_data('', dataset_name, use_pca=True, use_lda=True, use_smote=True)
# if 'CK+' in dataset_name:
#     file_output = 'ckplus' 
# elif 'RAFDB' in dataset_name:
#     file_output = 'rafdb' 
# elif 'FERP' in dataset_name:
#     file_output = 'ferp'
# elif 'JAFFE' in dataset_name:
#     file_output = 'jaffe'
# elif 'Bosphorus' in dataset_name:
#     file_output = 'bosphorus'
# else:
#     file_output = 'dataset'

# file_output = file_output + '_PCA_LDA_SMOTE.h5'

# save_path = os.path.join('datasets', dataset_name, file_output)

# with h5py.File(save_path, 'w') as dataset: 
#     for split in X.keys():
#         dataset.create_dataset(f'X_{split}', data=X[split])
#         dataset.create_dataset(f'y_{split}', data=y[split])

# del X, y

# print("Loading data with PCA and LDA...")

# X, y = load_data('', dataset_name, use_pca=True, use_lda=True, use_smote=False)
# if 'CK+' in dataset_name:
#     file_output = 'ckplus' 
# elif 'RAFDB' in dataset_name:
#     file_output = 'rafdb' 
# elif 'FERP' in dataset_name:
#     file_output = 'ferp'
# elif 'JAFFE' in dataset_name:
#     file_output = 'jaffe'
# elif 'Bosphorus' in dataset_name:
#     file_output = 'bosphorus'
# else:
#     file_output = 'dataset'

# file_output = file_output + '_PCA_LDA.h5'

# save_path = os.path.join('datasets', dataset_name, file_output)

# with h5py.File(save_path, 'w') as dataset: 
#     for split in X.keys():
#         dataset.create_dataset(f'X_{split}', data=X[split])
#         dataset.create_dataset(f'y_{split}', data=y[split])

# del X, y


print("Loading data with PCA and SMOTE...")

X, y = load_data('', dataset_name, use_pca=True, use_lda=False, use_smote=True)
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

file_output = file_output + '_PCA_SMOTE.h5'

save_path = os.path.join('datasets', dataset_name, file_output)

with h5py.File(save_path, 'w') as dataset: 
    for split in X.keys():
        dataset.create_dataset(f'X_{split}', data=X[split])
        dataset.create_dataset(f'y_{split}', data=y[split])

del X, y

# print("Loading data with PCA...")

# X, y = load_data('', dataset_name, use_pca=True, use_lda=False, use_smote=False)
# if 'CK+' in dataset_name:
#     file_output = 'ckplus' 
# elif 'RAFDB' in dataset_name:
#     file_output = 'rafdb' 
# elif 'FERP' in dataset_name:
#     file_output = 'ferp'
# elif 'JAFFE' in dataset_name:
#     file_output = 'jaffe'
# elif 'Bosphorus' in dataset_name:
#     file_output = 'bosphorus'
# else:
#     file_output = 'dataset'

# file_output = file_output + '_PCA.h5'

# save_path = os.path.join('datasets', dataset_name, file_output)

# with h5py.File(save_path, 'w') as dataset: 
#     for split in X.keys():
#         dataset.create_dataset(f'X_{split}', data=X[split])
#         dataset.create_dataset(f'y_{split}', data=y[split])

# del X, y



