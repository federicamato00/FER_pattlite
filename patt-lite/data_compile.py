import os
import cv2
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_unique_directory(base_path, dataset_name, dataset_name_2=None):
    dataset_dir = os.path.join(base_path, dataset_name, dataset_name_2) if dataset_name_2 else os.path.join(base_path, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    return dataset_dir

def load_data(
    path_prefix,
    dataset_name,
    test_size=0.2,
    val_size=0.1,
):
    X, y, subjects = [], [], []

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
        classNames = ['neutral', 'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
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
                                    frames_subject_path = os.path.join(frames_path, subject)
                                    frames_session_path = os.path.join(frames_subject_path, session)
                                    if os.path.isdir(frames_session_path):
                                        frames = sorted(os.listdir(frames_session_path))
                                        if len(frames) > 2:
                                            first_frame_path = os.path.join(frames_session_path, frames[0])
                                            
                                            if first_frame_path.endswith('.png'):
                                                image = cv2.imread(first_frame_path, cv2.IMREAD_COLOR)
                                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                                                X.append(image)
                                                y.append(classNames.index('neutral'))
                                                subjects.append(subject)
                                                if jump == False: 
                                                    for frame in frames[-2:]:
                                                        frame_path = os.path.join(frames_session_path, frame)
                                                        image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                                                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                                        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                                                        X.append(image)
                                                        y.append(emotion_label)
                                                        subjects.append(subject)
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
                                subjects.append(subject_folder)
    else:
        raise ValueError("Tipo di dataset non supportato: {}".format(dataset_name))

    X = np.array(X)
    y = np.array(y)
    subjects = np.array(subjects)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    mask = y != 2
    X = X[mask]
    y = y[mask]
    subjects = subjects[mask]
    
    unique_labels = np.unique(y)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y = np.array([label_mapping[label] for label in y])
    
    print(f"X shape after filtering: {X.shape}")
    print(f"y shape after filtering: {y.shape}")
    
    path_distribution = get_unique_directory('dataset_distribution', 'NO AUGMENTATION', dataset_name)
    dataset_dir = get_unique_directory('datasets', dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
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
    
    unique_subjects = np.unique(subjects)
    train_subjects, test_subjects = train_test_split(unique_subjects, test_size=test_size, random_state=42)
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_size / (1 - test_size), random_state=42)
    
    train_mask = np.isin(subjects, train_subjects)
    val_mask = np.isin(subjects, val_subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    return {'train': X_train, 'val': X_val, 'test': X_test}, {'train': y_train, 'val': y_val, 'test': y_test}, {'train': train_subjects, 'val': val_subjects, 'test': test_subjects}

dataset_name='CK+' # 'CK+', 'RAFDB', 'FERP', 'JAFFE', 'Bosphorus'
X, y, subjects = load_data('datasets/CK+', dataset_name)
if 'CK+' in dataset_name:
    file_output = 'ckplus_noP.h5'
elif 'RAFDB' in dataset_name:
    file_output = 'rafdb_noP.h5'
elif 'FERP' in dataset_name:
    file_output = 'ferp_noP.h5'
elif 'JAFFE' in dataset_name:
    file_output = 'jaffe_noP.h5'
elif 'Bosphorus' in dataset_name:
    file_output = 'bosphorus_noP.h5'
else:
    file_output = 'dataset_noP.h5'

file_save_path = os.path.join('datasets/NO AUGMENTATION', dataset_name, file_output)
with h5py.File(file_save_path, 'w') as dataset: 
    for split in X.keys():
        dataset.create_dataset(f'X_{split}', data=X[split])
        dataset.create_dataset(f'y_{split}', data=y[split])

del X, y

print(f"Dataset salvato in {file_save_path}")

# Stampa i soggetti unici per ogni set
print(f"Soggetti unici nel training set: {np.unique(subjects['train'])}")
print(f"Soggetti unici nel validation set: {np.unique(subjects['val'])}")
print(f"Soggetti unici nel test set: {np.unique(subjects['test'])}")