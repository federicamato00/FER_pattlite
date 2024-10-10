# Risultati e Grafici
 
Possiamo fare le seguenti osservazioni:

## Grafico 1: [confusion_matrix](./confusion_matrix.png)

L'immagine mostra una **matrice di confusione**, uno strumento utilizzato nel machine learning per valutare le prestazioni di un modello di classificazione. Ecco una spiegazione dei risultati:

- **Ottima precisione per la felicità**: Il modello ha identificato correttamente 42 istanze di felicità, dimostrando una forte capacità di riconoscere questa emozione.
- **Confusione tra emozioni**: Ci sono state alcune confusioni tra emozioni come rabbia e tristezza, con 5 istanze di tristezza erroneamente classificate come rabbia.
- **Performance generale**: La maggior parte delle etichette vere ha un alto numero di conteggi sulla diagonale, indicando classificazioni corrette. Tuttavia, ci sono alcune classi come 'disgusto' che hanno meno predizioni corrette.
- **Miglioramenti necessari**: Potrebbe essere utile migliorare la distinzione tra emozioni come rabbia e tristezza, e tra sorpresa e altre emozioni, dove si sono verificati alcuni errori di classificazione.


## Grafico 2: [Training e Validation plots](./training_validation_plots.png)

Ci sono due grafici:

1. **Training and Validation Accuracy**: Entrambe le linee mostrano un trend crescente, indicando che il modello sta migliorando la sua accuratezza sia sui dati di training che di validazione. La precisione di training è leggermente più alta rispetto a quella di validazione, suggerendo un buon apprendimento con un lieve rischio di sovra-addestramento.

2. **Training and Validation Loss**: Entrambe le linee mostrano un trend decrescente, il che è positivo. La perdita di training diminuisce più rapidamente rispetto alla perdita di validazione, indicando che il modello sta imparando bene dai dati di training ma potrebbe non generalizzare altrettanto bene sui dati di validazione.

Questi grafici suggeriscono che il modello sta migliorando nel tempo e possono essere considerati un buon punto di partenza. Tuttavia, potrebbe essere utile implementare tecniche di regolarizzazione per migliorare ulteriormente la generalizzazione. 


## [Parameters] (./parameters.txt)
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./parameters.txt). Per questa prova, si sono ottenuti i seguenti risultati:
- accuracy test set: 87.10%
- accuracy train set: 92.44%
- accuracy validation set: 91.72%

In questo caso, il BATCH_SIZE è stato impostato a 32
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



