import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Definisci i percorsi delle cartelle
base_dir = 'allcroppedimgs_lat,front,test'
frontal_dir = os.path.join(base_dir, 'frontal')
test_bosphorus_dir = os.path.join(base_dir, 'test_bosphorus')

# Crea le cartelle per training e validation
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Ottieni le sottocartelle (le emozioni)
emotions = os.listdir(frontal_dir)

# Dividi le immagini in training e validation
for emotion in emotions:
    emotion_dir = os.path.join(frontal_dir, emotion)
    images = os.listdir(emotion_dir)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
    
    # Crea le cartelle per ogni emozione in training e validation
    train_emotion_dir = os.path.join(train_dir, emotion)
    val_emotion_dir = os.path.join(val_dir, emotion)
    os.makedirs(train_emotion_dir, exist_ok=True)
    os.makedirs(val_emotion_dir, exist_ok=True)
    
    # Copia le immagini nelle rispettive cartelle
    for image in train_images:
        shutil.copy(os.path.join(emotion_dir, image), os.path.join(train_emotion_dir, image))
    for image in val_images:
        shutil.copy(os.path.join(emotion_dir, image), os.path.join(val_emotion_dir, image))

print("Dataset organizzato con successo!")

# Definisci i percorsi delle cartelle
base_dir = 'allcroppedimgs_lat,front,test'
test_bosphorus_dir = os.path.join(base_dir, 'test_bosphorus')
os.makedirs(test_bosphorus_dir, exist_ok=True)

# Ottieni le sottocartelle (le emozioni)
emotions = os.listdir(test_bosphorus_dir)

# Copia le immagini di test nelle rispettive cartelle
for emotion in emotions:
    emotion_dir = os.path.join(test_bosphorus_dir, emotion)
    test_emotion_dir = os.path.join(test_bosphorus_dir, emotion)
    os.makedirs(test_emotion_dir, exist_ok=True)
    
    images = os.listdir(emotion_dir)
    for image in images:
        shutil.copy(os.path.join(emotion_dir, image), os.path.join(test_emotion_dir, image))

print("Immagini di test organizzate con successo!")



# Crea i dataset di training e validation
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    frontal_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    frontal_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

# Crea il dataset di test
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_bosphorus_dir,
    image_size=(224, 224),
    batch_size=32
)

print("Dataset di training, validation e test creati con successo!")

# Funzione per creare un TFRecord
def create_tfrecord(dataset, filename):
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)

# Salva i dataset in formato TFRecord
create_tfrecord(train_dataset, os.path.join(base_dir, 'train.tfrecord'))
create_tfrecord(val_dataset, os.path.join(base_dir, 'val.tfrecord'))
create_tfrecord(test_dataset, os.path.join(base_dir, 'test.tfrecord'))

print("Dataset salvati con successo in formato TFRecord!")