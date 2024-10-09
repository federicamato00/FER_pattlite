import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa


def get_unique_directory(base_path, dataset_name):
    import os
    dataset_dir = os.path.join(base_path, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    return dataset_dir


def augment_images(images, labels, augmentations, target_count):
    augmented_images = []
    augmented_labels = []
    while len(augmented_images) < target_count:
        for image, label in zip(images, labels):
            augmented_image = augmentations(image=image)
            augmented_images.append(augmented_image)
            augmented_labels.append(label)
            if len(augmented_images) >= target_count:
                break
    return np.array(augmented_images), np.array(augmented_labels)

def load_data(
    path_prefix,
    dataset_name,
    test_size=0.2,
    val_size=0.1,
    use_augmentation=False,
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
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if not os.path.exists(path_distribution):
        os.makedirs(path_distribution)
    
    # Visualizza la distribuzione delle classi prima di augmentation
    plt.figure(figsize=(10, 5))
    plt.bar(np.unique(y), np.bincount(y))
    plt.title('Distribuzione delle classi')
    plt.xlabel('Classi')
    plt.ylabel('Numero di campioni')
    plt.xticks(ticks=np.unique(y), labels=[classNames[i] for i in np.unique(y)])
    plt.savefig(os.path.join(path_distribution, 'distribution_dataset.png'))
    
    if use_augmentation:
        # Controlla se le classi sono sbilanciate
        class_counts = np.bincount(y)
        max_count = np.max(class_counts)
        imbalance_threshold = 0.5  # Soglia per considerare una classe sbilanciata (50%)
        is_imbalanced = any(count < max_count * (1 - imbalance_threshold) for count in class_counts)
        
        if is_imbalanced:
            print("Le classi sono sbilanciate. Applicazione di data augmentation...")
            augmentations = iaa.Sequential([
                iaa.Fliplr(0.5),  # flip orizzontale
                iaa.Affine(rotate=(-20, 20)),  # rotazione
                iaa.Multiply((0.8, 1.2)),  # variazione di luminosità
                iaa.GaussianBlur(sigma=(0, 1.0))  # blur
            ])
            
            X_augmented, y_augmented = [], []
            for class_idx in np.unique(y):
                class_images = X[y == class_idx]
                class_labels = y[y == class_idx]
                target_count = max_count - len(class_images)
                if target_count > 0:
                    augmented_images, augmented_labels = augment_images(class_images, class_labels, augmentations, target_count)
                    X_augmented.append(augmented_images)
                    y_augmented.append(augmented_labels)
            
            X_augmented = np.concatenate(X_augmented, axis=0)
            y_augmented = np.concatenate(y_augmented, axis=0)
            
            X = np.concatenate((X, X_augmented), axis=0)
            y = np.concatenate((y, y_augmented), axis=0)
            
            # Visualizza la distribuzione delle classi dopo augmentation
            plt.figure(figsize=(10, 5))
            plt.bar(np.unique(y), np.bincount(y))
            plt.title('Distribuzione delle classi dopo data augmentation')
            plt.xlabel('Classi')
            plt.ylabel('Numero di campioni')
            plt.xticks(ticks=np.unique(y), labels=[classNames[i] for i in np.unique(y)])
            plt.savefig(os.path.join(path_distribution, 'dopo_augmentation.png'))
        else:
            print("Le classi sono bilanciate. Data augmentation non è necessaria.")

    # Split the data into train, val, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=42, stratify=y)
    val_size_adjusted = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp)
    
    return {'train': X_train, 'val': X_val, 'test': X_test}, {'train': y_train, 'val': y_val, 'test': y_test}

dataset_name='Bosphorus' # 'CK+', 'RAFDB', 'FERP', 'JAFFE', 'Bosphorus'

print("Loading data...")
X, y = load_data('', dataset_name, use_augmentation=True)
if 'CK+' in dataset_name:
    file_output = 'ckplus' 
elif 'RAFDB' in dataset_name:
    file_output = 'rafdb' 
elif 'FERP' in dataset_name:
    file_output = 'ferp'
elif 'JAFFE' in dataset_name:
    file_output = 'jaffe'
elif 'Bosphorus' in dataset_name:
    file_output = 'bosphorus_data_augmentation'
else:
    file_output = 'dataset'

file_output = file_output + '.h5'

with h5py.File(file_output, 'w') as dataset: 
    for split in X.keys():
        dataset.create_dataset(f'X_{split}', data=X[split])
        dataset.create_dataset(f'y_{split}', data=y[split])

del X, y

# print("Loading data with LDA...")


# X, y = load_data('', dataset_name, use_pca=False, use_lda=True, use_smote=False)
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

# file_output = file_output + '_LDA.h5'

# save_path = os.path.join('datasets', dataset_name, file_output)

# with h5py.File(save_path, 'w') as dataset: 
#     for split in X.keys():
#         dataset.create_dataset(f'X_{split}', data=X[split])
#         dataset.create_dataset(f'y_{split}', data=y[split])

# del X, y


# print("Loading data with LDA and SMOTE...")

# X, y = load_data('', dataset_name, use_pca=False, use_lda=True, use_smote=True)
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

# file_output = file_output + '_LDA_SMOTE.h5'

# save_path = os.path.join('datasets', dataset_name, file_output)

# with h5py.File(save_path, 'w') as dataset: 
#     for split in X.keys():
#         dataset.create_dataset(f'X_{split}', data=X[split])
#         dataset.create_dataset(f'y_{split}', data=y[split])

# del X, y


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


# print("Loading data with PCA and SMOTE...")

# X, y = load_data('', dataset_name, use_pca=True, use_lda=False, use_smote=True)
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

# file_output = file_output + '_PCA_SMOTE.h5'

# save_path = os.path.join('datasets', dataset_name, file_output)

# with h5py.File(save_path, 'w') as dataset: 
#     for split in X.keys():
#         dataset.create_dataset(f'X_{split}', data=X[split])
#         dataset.create_dataset(f'y_{split}', data=y[split])

# del X, y

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



