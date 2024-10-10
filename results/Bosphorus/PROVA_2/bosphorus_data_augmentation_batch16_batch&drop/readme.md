# Risultati e Grafici
 
Possiamo fare le seguenti osservazioni:

## Grafico 1: [confusion_matrix](./confusion_matrix.png)

L'immagine mostra una **matrice di confusione**, uno strumento utilizzato nel machine learning per valutare le prestazioni di un modello di classificazione. Ecco una spiegazione dei risultati:

- **Predizioni corrette**: La matrice mostra che le emozioni come **"neutral"** (41 predizioni corrette) e **"anger"** (40 predizioni corrette) sono state ben classificate dal modello.
- **Confusioni comuni**: Ci sono alcune confusioni evidenti, ad esempio, **"disgust"** è stato confuso con **"anger"** in 3 casi e **"fear"** con **"neutral"** in 5 casi.
- **Performance generale**: Il modello sembra avere una buona precisione per alcune emozioni, ma potrebbe migliorare nella distinzione tra emozioni simili come **"disgust"** e **"anger"**.

## Grafico 2: [Training e Validation plots](./training_validation_plots.png)

Ci sono due grafici:

1. **Training and Validation Accuracy**: Il grafico mostra un aumento costante dell'accuratezza sia per i dati di training che per quelli di validazione. Tuttavia, l'accuratezza del training è sempre leggermente superiore a quella della validazione, suggerendo che il modello sta imparando bene ma potrebbe esserci un leggero overfitting.

2. **Training and Validation Loss**: Il grafico della perdita mostra una diminuzione costante sia per il training che per la validazione. La perdita di training è inferiore a quella di validazione, il che è normale, ma se la differenza è troppo grande, potrebbe indicare overfitting.

Questi grafici indicano che il modello sta migliorando nel tempo, ma è importante monitorare l'overfitting per garantire che il modello generalizzi bene su dati non visti. 

## [Parameters] (./parameters.txt)
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./parameters.txt). Per questa prova, si sono ottenuti i seguenti risultati:
- accuracy test set: 88.03%
- accuracy train set: 85.71%
- accuracy validation set: 91.40%

In questo caso, il BATCH_SIZE è stato impostato a 16.
Nella patch_extraction sono stati usati i seguenti parametri:
patch_extraction = tf.keras.Sequential([
    tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'), 
    tf.keras.layers.BatchNormalization(),  
    tf.keras.layers.Dropout(dropout_rate),  # Aggiungi dropout
    tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'), 
    tf.keras.layers.BatchNormalization(),  
    tf.keras.layers.Dropout(dropout_rate),  # Aggiungi dropout
    tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(best_hps['l2_reg']))
], name='patch_extraction')

La patch_extraction è definita come una sequenza di livelli convoluzionali, batch normalization e dropout. Ecco una spiegazione dettagliata di ciascun componente:

- SeparableConv2D:

    - Convoluzione Separable: Utilizza convoluzioni separabili in profondità per ridurre il numero di parametri e migliorare l'efficienza computazionale. Questo tipo di convoluzione separa la convoluzione spaziale e la convoluzione in profondità, riducendo il costo computazionale.
    - Kernel Size e Strides: Il primo livello ha un kernel di dimensione 4x4 con stride di 4, mentre il secondo livello ha un kernel di dimensione 2x2 con stride di 2. Questi parametri determinano la dimensione delle patch estratte e la riduzione della risoluzione spaziale.
- BatchNormalization:
    - Normalizzazione del Batch: Normalizza le attivazioni di ciascun batch per stabilizzare e accelerare l'addestramento. Aiuta a ridurre il problema del covariate shift interno.

- Dropout:
   - Dropout: Aggiunge regolarizzazione al modello disattivando casualmente una frazione delle unità durante l'addestramento. Questo aiuta a prevenire l'overfitting.
   
- Conv2D:
    - Convoluzione 2D: Applica una convoluzione standard con kernel di dimensione 1x1 per ridurre ulteriormente la dimensionalità e combinare le caratteristiche estratte dai livelli precedenti.
    - Kernel Regularizer: Utilizza la regolarizzazione L2 per penalizzare i pesi grandi e prevenire l'overfitting.
    Con un batch_size per il modello settato a 8

1. Scopo della patch_extraction
    -Riduzione della Dimensionalità: Riduce la risoluzione spaziale delle immagini, permettendo al modello di concentrarsi su caratteristiche più astratte e di livello superiore.
    -Estrazione delle Caratteristiche: Evidenzia pattern rilevanti nelle immagini, come bordi, texture e forme, che sono utili per la classificazione.
    -Efficienza Computazionale: Utilizza convoluzioni separabili per ridurre il numero di parametri e il costo computazionale, rendendo il modello più efficiente.

La patch_extraction è una componente cruciale del tuo modello che aiuta a trasformare le immagini di input in rappresentazioni più compatte e informative, facilitando il compito di classificazione delle espressioni facciali.



