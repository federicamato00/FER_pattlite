import os
import cv2
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

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
    
    # Split the data into train, val, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=42, stratify=y)
    val_size_adjusted = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp)
    
    return {'train': X_train, 'val': X_val, 'test': X_test}, {'train': y_train, 'val': y_val, 'test': y_test}


dataset_name='CK+' # 'CK+', 'RAFDB', 'FERP', 'JAFFE', 'Bosphorus'
X, y = load_data('', dataset_name)
if 'CK+' in dataset_name:
    file_output = 'ckplus.h5'
elif 'RAFDB' in dataset_name:
    file_output = 'rafdb.h5'
elif 'FERP' in dataset_name:
    file_output = 'ferp.h5'
elif 'JAFFE' in dataset_name:
    file_output = 'jaffe.h5'
elif 'Bosphorus' in dataset_name:
    file_output = 'bosphorus.h5'
else:
    file_output = 'dataset.h5'


with h5py.File(file_output, 'w') as dataset: 
    for split in X.keys():
        dataset.create_dataset(f'X_{split}', data=X[split])
        dataset.create_dataset(f'y_{split}', data=y[split])

del X, y