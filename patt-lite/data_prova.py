import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa


def get_unique_directory(base_path, dataset_name, dataset_name_2=None):
    dataset_dir = os.path.join(base_path, dataset_name, dataset_name_2) if dataset_name_2 else os.path.join(base_path, dataset_name)
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
    additional_images_per_class=100
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
        classNames = ['neutral', 'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']  ## 7 classi
        # classNames = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']  ## 8 classi
    else:
        classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

    PATH = os.path.join(path_prefix, dataset_name)
    if dataset_name == 'CK+':
        emotion_path = os.path.join(PATH, 'Emotion')
        frames_path = os.path.join(PATH, 'cohn-kanade-images')
        for subject in os.listdir(emotion_path):
            subject_path = os.path.join(emotion_path, subject)
            if os.path.isdir(subject_path):
                for session in os.listdir(subject_path):
                    session_path = os.path.join(subject_path, session)
                    if os.path.isdir(session_path):
                        for file in os.listdir(session_path):
                            jump = False
                            if file.endswith('.txt'):
                                file_path = os.path.join(session_path, file)
                                if os.path.getsize(file_path) == 0:
                                    jump = True
                                    continue
                                with open(file_path, 'r') as f:
                                    emotion_label = int(float(f.readline().strip()))
                                
                                if emotion_label != 2:  # Filtra le espressioni 'contempt'
                                    # Trova i frame corrispondenti nella cartella dei frames
                                    frames_subject_path = os.path.join(frames_path, subject)
                                    frames_session_path = os.path.join(frames_subject_path, session)
                                    if os.path.isdir(frames_session_path):
                                        frames = sorted(os.listdir(frames_session_path))
                                        if len(frames) > 2:
                                            # Primo frame come "neutro"
                                            first_frame_path = os.path.join(frames_session_path, frames[0])
                                            
                                            if first_frame_path.endswith('.png'):
                                                image = cv2.imread(first_frame_path, cv2.IMREAD_COLOR)
                                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                                                X.append(image)
                                                y.append(classNames.index('neutral'))
                                                # Ultimi due frame come emotion_label
                                                if jump == False: 
                                                    for frame in frames[-2:]:
                                                        frame_path = os.path.join(frames_session_path, frame)
                                                        image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                                                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                                        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                                                        X.append(image)
                                                        y.append(emotion_label)
                                                else:
                                                    jump = False

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
    
    # Filtra le etichette "2" e riscalare le etichette rimanenti
    mask = y != 2
    X = X[mask]
    y = y[mask]
    
    # Riscalare le etichette rimanenti
    unique_labels = np.unique(y)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y = np.array([label_mapping[label] for label in y])
    
    print(f"X shape after filtering: {X.shape}")
    print(f"y shape after filtering: {y.shape}")
    
    path_distribution = get_unique_directory('dataset_distribution', dataset_name, 'CKplus_numClasses7')
    # Creare le directory per salvare i dati e i grafici
    dataset_dir = get_unique_directory('datasets', dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Visualizza la distribuzione delle classi prima di augmentation
    plt.figure(figsize=(10, 5))
    plt.bar(np.unique(y), np.bincount(y))
    plt.title('Distribuzione delle classi')
    plt.xlabel('Classi')
    plt.ylabel('Numero di campioni')
    if 'CK+' in dataset_name:
        classNames = {emotion: idx for idx, emotion in enumerate(classNames)}
        classNames = {idx: emotion for idx, emotion in enumerate(classNames)}
    print(classNames)
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
            # '''prima prova, salvato in boss_data_augmentation.h5'''
            # augmentations = iaa.Sequential([
            #     iaa.Fliplr(0.5),  # flip orizzontale
            #     iaa.Affine(rotate=(-20, 20)),  # rotazione
            #     iaa.Multiply((0.8, 1.2)),  # variazione di luminosità
            #     iaa.GaussianBlur(sigma=(0, 1.0))  # blur
            # ])

        #     '''seconda prova, salvato in boss_data_augmentation_2.h5'''
        #     augmentations = iaa.Sequential([
        #     iaa.Fliplr(0.5),  # flip orizzontale
        #     iaa.Affine(rotate=(-20, 20)),  # rotazione
        #     iaa.Multiply((0.8, 1.2)),  # variazione di luminosità
        #     iaa.GaussianBlur(sigma=(0, 1.0)),  # blur
        #     iaa.TranslateX(percent=(-0.2, 0.2)),  # traslazione orizzontale
        #     iaa.TranslateY(percent=(-0.2, 0.2)),  # traslazione verticale
        #     iaa.ScaleX((0.8, 1.2)),  # zoom orizzontale
        #     iaa.ScaleY((0.8, 1.2)),  # zoom verticale
        #     iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # rumore gaussiano
        #     iaa.Cutout(nb_iterations=1, size=0.2, squared=True),  # cutout
        #     iaa.ElasticTransformation(alpha=50, sigma=5)  # trasformazione elastica
        # ])
            
        #     '''terza prova, salvato in boss_data_augmentation_3.h5'''
        #     augmentations = iaa.Sequential([
        #     iaa.Fliplr(0.5),  # flip orizzontale
        #     iaa.Affine(rotate=(-20, 20)),  # rotazione
        #     iaa.Multiply((0.8, 1.2)),  # variazione di luminosità
        #     iaa.GaussianBlur(sigma=(0, 1.0)),  # blur
        #     iaa.TranslateX(percent=(-0.2, 0.2)),  # traslazione orizzontale
        #     iaa.TranslateY(percent=(-0.2, 0.2)),  # traslazione verticale
        #     iaa.ScaleX((0.8, 1.2)),  # zoom orizzontale
        #     iaa.ScaleY((0.8, 1.2)),  # zoom verticale
        #     iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # rumore gaussiano
        # ])

            '''quinta prova, salvato in boss_data_augmentation_5.h5, con augmentation iniziale + aggiunta nuovi dati'''
            augmentations = iaa.Sequential([
                iaa.Fliplr(0.5),  # flip orizzontale
                iaa.Affine(rotate=(-20, 20)),  # rotazione
                iaa.Multiply((0.8, 1.2)),  # variazione di luminosità
                iaa.GaussianBlur(sigma=(0, 1.0))  # blur
            ])

            
            X_augmented, y_augmented = [], []
            # Trova il numero massimo di campioni per classe
            unique_classes, class_counts = np.unique(y, return_counts=True)
            max_count = np.max(class_counts)
            for class_idx in unique_classes:
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
            plt.savefig(os.path.join(path_distribution, 'dopo_augmentation_5.png'))
        else:
            print("Le classi sono bilanciate. Data augmentation non è necessaria.")

            
        '''prova ad aumentare il numero di campioni per classe'''
        '''salvato in boss_data_augmentation_5.h5'''
        # Aumenta ulteriormente il numero di dati per ogni classe
        X_augmented, y_augmented = [], []
        for class_idx in unique_classes:
            class_images = X[y == class_idx]
            class_labels = y[y == class_idx]
            target_count = additional_images_per_class
            augmented_images, augmented_labels = augment_images(class_images, class_labels, augmentations, target_count)
            X_augmented.append(augmented_images)
            y_augmented.append(augmented_labels)
        
        X_augmented = np.concatenate(X_augmented, axis=0)
        y_augmented = np.concatenate(y_augmented, axis=0)
        
        X = np.concatenate((X, X_augmented), axis=0)
        y = np.concatenate((y, y_augmented), axis=0)
        print("Ulteriore aumento del numero di campioni per classe completato.")
        # Visualizza la distribuzione delle classi dopo ulteriore aumento
        plt.figure(figsize=(10, 5))
        plt.bar(np.unique(y), np.bincount(y))
        plt.title('Distribuzione delle classi dopo ulteriore aumento')
        plt.xlabel('Classi')
        plt.ylabel('Numero di campioni')
        plt.xticks(ticks=np.unique(y), labels=[classNames[i] for i in np.unique(y)])
        plt.savefig(os.path.join(path_distribution, 'dopo_ulteriore_aumento_2.png'))
   

    # Split the data into train, val, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=42, stratify=y)
    val_size_adjusted = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp)
    
    return {'train': X_train, 'val': X_val, 'test': X_test}, {'train': y_train, 'val': y_val, 'test': y_test}

dataset_name='CK+' # 'CK+', 'RAFDB', 'FERP', 'JAFFE', 'Bosphorus'

print("Loading data...")
X, y = load_data('datasets/CK+', dataset_name, use_augmentation=True)
if 'CK+' in dataset_name:
    file_output = 'ckplus_data_augmentation' 
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

file_path_save = os.path.join('datasets', dataset_name, 'CKplus_numClasses7',file_output)
with h5py.File(file_path_save, 'w') as dataset: 
    for split in X.keys():
        dataset.create_dataset(f'X_{split}', data=X[split])
        dataset.create_dataset(f'y_{split}', data=y[split])

del X, y

print(f"Dataset salvato in {file_path_save}") 
