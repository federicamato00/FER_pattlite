# Risultati e Grafici
 
 Possiamo fare le seguenti osservazioni:

## Grafico 1: [confusion_matrix](./bosphorus_data_augmentation_3/confusion_matrix.png)

La confusion matrix mostra le prestazioni di un modello di classificazione su un set di dati di test. Le righe rappresentano le etichette vere, mentre le colonne rappresentano le etichette predette dal modello.

1. **Predizioni Corrette**: Le celle diagonali (dall'angolo in alto a sinistra all'angolo in basso a destra) indicano le predizioni corrette. Ad esempio, il modello ha predetto correttamente 19 casi di "anger", 20 di "disgust", 13 di "fear", 29 di "happiness", 19 di "sadness", 23 di "surprise" e 33 di "neutral".

2. **Errori di Predizione**: Le celle fuori dalla diagonale mostrano le predizioni errate. Ad esempio, "fear" è stato spesso confuso con "surprise" (17 casi) e "happiness" è stato confuso con "sadness" (3 casi).

3. **Classi Ben Predette**: "Neutral" è la classe meglio predetta con 33 predizioni corrette, seguita da "happiness" con 29 predizioni corrette.

4. **Classi Confuse**: "Fear" e "surprise" mostrano una confusione significativa, con "fear" spesso predetto come "surprise".

- **Overfitting**: Se le predizioni corrette sui dati di addestramento sono molto più alte rispetto ai dati di test, potrebbe esserci un problema di overfitting.
- **Bilanciamento del Dataset**: Se alcune classi hanno molte più istanze corrette rispetto ad altre, potrebbe essere utile bilanciare il dataset per migliorare le prestazioni del modello.


## Grafico 2: [Training e Validation plots](./bosphorus_data_augmentation_3/training_validation_plots.png)

Ci sono due grafici:
1. Training e Validation Accuracy
- **Curva Blu (Training Accuracy)**: Mostra l'accuratezza del modello sui dati di addestramento. La curva sale costantemente, indicando che il modello sta imparando bene dai dati di addestramento.
- **Curva Arancione (Validation Accuracy)**: Mostra l'accuratezza del modello sui dati di validazione. Anche questa curva sale, ma con alcune fluttuazioni. Questo suggerisce che il modello sta migliorando anche sui dati di validazione, ma con qualche variazione.

2. Training and Validation Loss
- **Curva Blu (Training Loss)**: Mostra la perdita del modello sui dati di addestramento. La curva scende, indicando che il modello sta riducendo l'errore sui dati di addestramento.
- **Curva Arancione (Validation Loss)**: Mostra la perdita del modello sui dati di validazione. Anche questa curva scende, ma mostra più variabilità rispetto alla curva di training loss. Questo potrebbe indicare che il modello sta imparando, ma potrebbe esserci qualche problema di overfitting.

3. Osservazioni
    1. **Overfitting**: La differenza tra le curve di training e validation suggerisce che il modello potrebbe essere overfitting, cioè sta imparando troppo bene i dettagli dei dati di addestramento e non generalizza bene ai dati nuovi.
    2. **Stabilità**: Le fluttuazioni nella curva di validation loss indicano che il modello potrebbe beneficiare di ulteriori tecniche di regolarizzazione o di un maggior numero di dati di addestramento.


## [Parameters]
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./bosphorus_data_augmentation_3/parameters.txt). Per questa prima prova, si sono ottenuti i seguenti risultati:
- accuracy test set: 65.27%
- accuracy train set: 83.54%
- accuracy validation set: 64.08%


