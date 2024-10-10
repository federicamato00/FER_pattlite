# Risultati e Grafici

Questa cartella contiene varie prove e vari grafici dei risultati ottenuti usando diverse combinazioni di dataset e/o parametri per il modello da addestrare.


## Grafico 1: [confusion_matrix](./confusion_matrix.png)

1. **Classi e Predizioni**: La matrice di confusione mostra le predizioni del modello per sette classi di emozioni: **anger (rabbia)**, **disgust (disgusto)**, **fear (paura)**, **happiness (felicità)**, **sadness (tristezza)**, **surprise (sorpresa)** e **neutral (neutro)**. Le righe rappresentano le etichette vere, mentre le colonne rappresentano le etichette predette.

2. **Predizioni Corrette**: Le celle diagonali (dall'angolo in alto a sinistra all'angolo in basso a destra) indicano le predizioni corrette. Ad esempio, il modello ha predetto correttamente 30 casi di "anger", 27 di "fear", 34 di "happiness", 31 di "sadness", 27 di "surprise" e 34 di "neutral".

3. **Errori di Predizione**: Le celle fuori dalla diagonale mostrano le predizioni errate. Ad esempio, c'è un caso in cui "sadness" è stato predetto come "anger" e un caso in cui "neutral" è stato predetto come "anger".


## Grafico 2: [Training e Validation plots](./training_validation_plots.png)
Ci sono due sottografici: 
1.  Training and Validation Accuracy
    - **Curva Blu (Training Accuracy)**: Mostra l'accuratezza del modello sui dati di addestramento. La curva sale costantemente, indicando che il modello sta imparando bene dai dati di addestramento.
    - **Curva Arancione (Validation Accuracy)**: Mostra l'accuratezza del modello sui dati di validazione. Anche questa curva sale, ma rimane sempre sotto la curva di training, suggerendo che il modello potrebbe non generalizzare perfettamente ai dati non visti.

2. Training and Validation Loss
    - **Curva Blu (Training Loss)**: Mostra la perdita del modello sui dati di addestramento. La curva scende, indicando che il modello sta riducendo l'errore sui dati di addestramento.
    - **Curva Arancione (Validation Loss)**: Mostra la perdita del modello sui dati di validazione. Anche questa curva scende, ma con più fluttuazioni rispetto alla curva di training, suggerendo possibili problemi di overfitting.

### Osservazioni
1. **Overfitting**: La differenza tra le curve di training e validation suggerisce che il modello potrebbe essere overfitting, cioè sta imparando troppo bene i dettagli dei dati di addestramento e non generalizza bene ai dati nuovi.
2. **Stabilità**: Le fluttuazioni nella curva di validation loss indicano che il modello potrebbe beneficiare di ulteriori tecniche di regolarizzazione o di un maggior numero di dati di addestramento.



## [Parameters](./parameters.txt)
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./parameters.txt). Per questa prima prova, si sono ottenuti i seguenti risultati:
- accuracy test set: 90.38%
- accuracy train set: 93.15%
- accuracy validation set: 88.45%


