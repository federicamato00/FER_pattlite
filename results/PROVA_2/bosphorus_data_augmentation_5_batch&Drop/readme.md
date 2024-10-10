# Risultati e Grafici
 
Possiamo fare le seguenti osservazioni:

## Grafico 1: [confusion_matrix](./bosphorus_data_augmentation_5_batch&Drop/confusion_matrix.png)

L'immagine mostra una **matrice di confusione**, uno strumento utilizzato nel machine learning per valutare le prestazioni di un modello di classificazione. Ecco una spiegazione dei risultati:

- **Predizioni corrette**: I numeri sulla diagonale principale (dall'angolo in alto a sinistra all'angolo in basso a destra) rappresentano le predizioni corrette. Ad esempio, l'emozione "anger" è stata correttamente predetta 15 volte.
- **Predizioni errate**: I numeri fuori dalla diagonale principale indicano le predizioni errate. Ad esempio, "anger" è stata erroneamente predetta come "disgust" 3 volte.
- **Categorie di emozioni**: Le emozioni considerate sono "anger", "disgust", "fear", "happiness", "sadness", "surprise" e "neutral".
- **Performance complessiva**: La matrice mostra come il modello confonde alcune emozioni con altre, fornendo un'idea di dove migliorare.


## Grafico 2: [Training e Validation plots](./bosphorus_data_augmentation_5_batch&Drop/training_validation_plots.png)

Ci sono due grafici:
1. Training and Validation Accuracy
    - **Training Accuracy (linea blu)**: Mostra come l'accuratezza del modello sui dati di addestramento cambia nel tempo. Si osserva un trend crescente con alcune fluttuazioni.
    - **Validation Accuracy (linea arancione)**: Mostra l'accuratezza del modello sui dati di validazione. Anche qui si vede un trend crescente, ma con più fluttuazioni rispetto alla linea blu.

2. Training and Validation Loss
    - **Training Loss (linea blu)**: Indica la perdita del modello sui dati di addestramento. Un trend decrescente suggerisce che il modello sta migliorando.
    - **Validation Loss (linea arancione)**: Mostra la perdita sui dati di validazione. Anche questa linea decresce, ma con fluttuazioni, indicando come il modello si comporta su dati non visti.

In sintesi, i risultati mostrano che il modello sta migliorando, ma le fluttuazioni nella validazione suggeriscono che potrebbe essere necessario un ulteriore tuning per migliorare la generalizzazione. 

## [Parameters]
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./bosphorus_data_augmentation_5_batch&Drop/parameters.txt). Per questa prova, si sono ottenuti i seguenti risultati:
- accuracy test set: 60.52%
- accuracy train set: 69.00%
- accuracy validation set: 63.64%

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



