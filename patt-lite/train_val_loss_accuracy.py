import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
# Carica i modelli finali
model1 = tf.keras.models.load_model('final_models/PROVA_BEST_PARAMETERS/Bosphorus_data_augmentation_5_best_parameters_moreEpoch/Bosphorus_model.keras')
model2 = tf.keras.models.load_model('final_models/PROVA_BEST_PARAMETERS/Bosphorus_data_augmentation_5_500Epoch/Bosphorus_model.keras')
dataset_name='Bosphorus'

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


X_train, y_train, X_valid, y_valid, X_test, y_test = load_images_and_labels('processed_bosphorus_5.h5')

# Valuta i modelli sui dati di addestramento e validazione
train_loss1, train_acc1 = model1.evaluate(X_train, y_train)
val_loss1, val_acc1 = model1.evaluate(X_valid, y_valid)

train_loss2, train_acc2 = model2.evaluate(X_train, y_train)
val_loss2, val_acc2 = model2.evaluate(X_valid, y_valid)

# Plotta i risultati finali
labels = ['Model 1', 'Model 2']
train_accuracies = [train_acc1, train_acc2]
val_accuracies = [val_acc1, val_acc2]
train_losses = [train_loss1, train_loss2]
val_losses = [val_loss1, val_loss2]

x = range(len(labels))

plt.figure(figsize=(12, 6))

# Grafico di Accuratezza
plt.subplot(1, 2, 1)
plt.bar(x, train_accuracies, width=0.4, label='Training Accuracy', align='center')
plt.bar(x, val_accuracies, width=0.4, label='Validation Accuracy', align='edge')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.xticks(x, labels)
plt.legend()
plt.title('Training and Validation Accuracy')

# Grafico di Perdita
plt.subplot(1, 2, 2)
plt.bar(x, train_losses, width=0.4, label='Training Loss', align='center')
plt.bar(x, val_losses, width=0.4, label='Validation Loss', align='edge')
plt.xlabel('Models')
plt.ylabel('Loss')
plt.xticks(x, labels)
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 4))

# Grafico di Accuratezza
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Grafico di Perdita
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')


plt.savefig(os.path.join(unique_dir, 'training_validation_plots.png'))
plt.show()
