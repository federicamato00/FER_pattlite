# Risultati e Grafici
 
Possiamo fare le seguenti osservazioni:

## Grafico 1: [confusion_matrix](./confusion_matrix.png)

L'immagine mostra una **matrice di confusione**, uno strumento utilizzato nel machine learning per valutare le prestazioni di un modello di classificazione. Ecco una spiegazione dei risultati:

- **Predizioni corrette**: I numeri sulla diagonale principale (dall'angolo in alto a sinistra all'angolo in basso a destra) rappresentano le predizioni corrette. Ad esempio, l'emozione "anger" è stata correttamente predetta 15 volte.
- **Predizioni errate**: I numeri fuori dalla diagonale principale indicano le predizioni errate. Ad esempio, "anger" è stata erroneamente predetta come "disgust" 3 volte.
- **Categorie di emozioni**: Le emozioni considerate sono "anger", "disgust", "fear", "happiness", "sadness", "surprise" e "neutral".
- **Performance complessiva**: La matrice mostra come il modello confonde alcune emozioni con altre, fornendo un'idea di dove migliorare.


## Grafico 2: [Training e Validation plots](./training_validation_plots.png)

Ci sono due grafici:
1. Training and Validation Accuracy
    - **Training Accuracy (linea blu)**: Mostra come l'accuratezza del modello sui dati di addestramento cambia nel tempo. Si osserva un trend crescente con alcune fluttuazioni.
    - **Validation Accuracy (linea arancione)**: Mostra l'accuratezza del modello sui dati di validazione. Anche qui si vede un trend crescente, ma con più fluttuazioni rispetto alla linea blu.

2. Training and Validation Loss
    - **Training Loss (linea blu)**: Indica la perdita del modello sui dati di addestramento. Un trend decrescente suggerisce che il modello sta migliorando.
    - **Validation Loss (linea arancione)**: Mostra la perdita sui dati di validazione. Anche questa linea decresce, ma con fluttuazioni, indicando come il modello si comporta su dati non visti.

In sintesi, i risultati mostrano che il modello sta migliorando, ma le fluttuazioni nella validazione suggeriscono che potrebbe essere necessario un ulteriore tuning per migliorare la generalizzazione. 

## [Parameters](./parameters.txt) 
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./parameters.txt). Per questa prima prova, si sono ottenuti i seguenti risultati:
- accuracy test set: 60.52%
- accuracy train set: 69.00%
- accuracy validation set: 63.64%



