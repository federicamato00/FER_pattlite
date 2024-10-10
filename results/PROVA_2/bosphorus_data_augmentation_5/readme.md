# Risultati e Grafici
 
Possiamo fare le seguenti osservazioni:

## Grafico 1: [confusion_matrix](./bosphorus_data_augmentation_4/confusion_matrix.png)

Dalla matrice di confusione che hai fornito, puoi dedurre le seguenti informazioni sul tuo modello di classificazione:

- **Predizioni corrette**: I numeri sulla diagonale principale (dall'angolo in alto a sinistra all'angolo in basso a destra) rappresentano le predizioni corrette per ciascuna emozione. Ad esempio, "anger" è stata correttamente predetta 40 volte, "disgust" 42 volte, e così via.
- **Predizioni errate**: I numeri fuori dalla diagonale principale indicano le predizioni errate. Ad esempio, "anger" è stata erroneamente predetta come "disgust" una volta.
- **Performance complessiva**: La matrice mostra che il modello ha una buona accuratezza per alcune emozioni come "happiness" e "neutral", ma potrebbe avere difficoltà a distinguere tra emozioni simili come "anger" e "disgust".

Questa analisi ti aiuta a capire dove il modello sta performando bene e dove potrebbe necessitare di miglioramenti. 

## Grafico 2: [Training e Validation plots](./bosphorus_data_augmentation_3/training_validation_plots.png)

Ci sono due grafici:

1. Training and Validation Accuracy
    - **Training Accuracy (linea blu)**: Mostra come l'accuratezza del modello migliora durante l'addestramento. La linea tende verso l'alto, indicando che il modello sta imparando bene dai dati di addestramento.
    - **Validation Accuracy (linea arancione)**: Rappresenta l'accuratezza del modello sui dati di validazione. Anche questa linea tende verso l'alto, ma con alcune fluttuazioni, suggerendo che il modello generalizza bene ma potrebbe avere qualche variabilità.

2. Training and Validation Loss
    - **Training Loss (linea blu)**: Mostra la perdita del modello durante l'addestramento. La linea scende, indicando che il modello sta riducendo l'errore sui dati di addestramento.
    - **Validation Loss (linea arancione)**: Rappresenta la perdita sui dati di validazione. Anche questa linea scende, ma con alcune fluttuazioni, suggerendo che il modello sta migliorando ma potrebbe esserci un po' di overfitting.

3. Interpretazione dei Risultati
    - **Tendenza Positiva**: L'aumento dell'accuratezza e la diminuzione della perdita indicano che il modello sta migliorando.
    - **Fluttuazioni**: Le fluttuazioni nelle linee di validazione potrebbero indicare variabilità nei dati o potenziale overfitting. Potrebbe essere utile monitorare queste fluttuazioni e considerare tecniche di regolarizzazione.


## [Parameters]
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./bosphorus_data_augmentation_5/parameters.txt). Per questa prima prova, si sono ottenuti i seguenti risultati:

accuracy test set: 92.88%
accuracy train set: 94.71%
accuracy validation set: 93.83%



