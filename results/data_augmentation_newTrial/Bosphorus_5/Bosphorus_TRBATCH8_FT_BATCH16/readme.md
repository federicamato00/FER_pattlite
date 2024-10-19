
### [Confusion Matrix](confusion_matrix.png)

- **Descrizione Generale**: La matrice di confusione mostra le prestazioni di un modello di classificazione su un set di dati di test. Le etichette includono emozioni come rabbia, disgusto, paura, felicità, tristezza, sorpresa e neutrale.
- **Predizioni Corrette**: La diagonale principale (dall'angolo in alto a sinistra a quello in basso a destra) rappresenta le predizioni corrette. I valori non nulli sono:
  - Rabbia: 43
  - Disgusto: 44
  - Paura: 44
  - Felicità: 44
  - Tristezza: 44
  - Sorpresa: 45
  - Neutrale: 45
- **Predizioni Errate**: Tutte le altre celle rappresentano predizioni errate e mostrano valori pari a zero, indicando che non ci sono stati errori di classificazione in questo esempio.

### Interpretazione
- **Alta Accuratezza**: La presenza di valori non nulli solo sulla diagonale principale suggerisce che il modello ha classificato correttamente tutte le istanze del set di dati di test.
- **Assenza di Errori**: L'assenza di valori nelle celle fuori dalla diagonale principale indica che non ci sono stati errori di classificazione, il che è un risultato eccellente.

## Grafici Accuracy and Loss

### [Grafico 1: Training and Validation Accuracy](training_validation_plots.png)

- **Andamento**: Entrambe le linee, sia per l'accuratezza del training (blu) che per quella della validazione (arancione), mostrano un rapido aumento iniziale seguito da una fase di stabilizzazione.
- **Interpretazione**: Questo indica che il modello ha migliorato rapidamente la sua capacità di prevedere correttamente i risultati all'inizio del training, per poi raggiungere un plateau. La stabilizzazione suggerisce che il modello ha raggiunto un buon livello di accuratezza.

### [Grafico 2: Training and Validation Loss](./training_validation_plots.png)

- **Andamento**: Le linee del loss per il training (blu) e la validazione (arancione) diminuiscono rapidamente all'inizio e poi si stabilizzano.
- **Interpretazione**: Questo suggerisce che l'errore del modello è diminuito rapidamente all'inizio del training, per poi raggiungere un livello di miglioramento minimo. La stabilizzazione del loss indica che il modello ha raggiunto un punto in cui ulteriori miglioramenti sono marginali.


### [Parameters](./parameters.txt)

accuracy test set: 0.9935275316238403
accuracy train set: 0.9976798295974731
accuracy validation set: 0.9821428656578064
loss test set: 0.04559168964624405
loss train set: 0.01611887663602829
loss validation set: 0.12412986159324646

F1 Score: 0.9935262725894791
Precision: 0.9936697778403139
Recall: 0.9935275080906149
AUC-ROC: 0.9999389673917977

## Analisi dei Risultati 
1. **Accuracy and Loss**

    ### Accuratezza
    - **Train set**: 0.9977
    - **Validation set**: 0.9821
    - **Test set**: 0.9935

    ### Loss
    - **Train set**: 0.0161
    - **Validation set**: 0.1241
    - **Test set**: 0.0456

    ### Interpretazione
    1. **Accuratezza**:
    - **Train set**: Un'accuratezza del 99.77% indica che il modello si comporta estremamente bene sui dati di addestramento.
    - **Validation set**: Un'accuratezza del 98.21% sui dati di validazione è molto alta e suggerisce che il modello generalizza bene.
    - **Test set**: Un'accuratezza del 99.35% sui dati di test è eccellente e conferma che il modello mantiene prestazioni elevate anche su dati non visti.

    2. **Loss**:
    - **Train set**: Un valore di loss molto basso (0.0161) sui dati di addestramento indica che il modello ha appreso molto bene i pattern nei dati.
    - **Validation set**: Un valore di loss di 0.1241 sui dati di validazione è accettabile e suggerisce che il modello non soffre di overfitting significativo.
    - **Test set**: Un valore di loss di 0.0456 sui dati di test è molto basso, confermando che il modello generalizza bene.

    ### Conclusione
    Il modello sembra performare in modo eccellente, con un'accuratezza molto alta su tutti i set di dati e valori di loss molto bassi. Questo indica che il modello è ben bilanciato e non soffre di overfitting.


2. **Additional Metrics**


    ### Metriche di Valutazione
    - **F1 Score**: 0.9935
    - **Precision**: 0.9937
    - **Recall**: 0.9935
    - **AUC-ROC**: 0.9999

    1. **Interpretazione**
    1. **F1 Score**:
    - Un F1 Score di 0.9935 indica un eccellente equilibrio tra precisione e recall. È una metrica utile quando hai bisogno di bilanciare entrambi gli aspetti, specialmente in presenza di classi sbilanciate.

    2. **Precision**:
    - Una precisione di 0.9937 significa che il 99.37% delle previsioni positive del modello sono corrette. Questo è importante quando il costo di un falso positivo è alto.

    3. **Recall**:
    - Un recall di 0.9935 indica che il modello è in grado di identificare correttamente il 99.35% delle istanze positive. Questo è cruciale quando il costo di un falso negativo è alto.

    4. **AUC-ROC**:
    - Un valore di AUC-ROC di 0.9999 è eccezionale e suggerisce che il modello è estremamente efficace nel distinguere tra le classi positive e negative. Un valore vicino a 1 indica un modello quasi perfetto.

    ### Conclusione
    Questi risultati confermano ulteriormente che il tuo modello è altamente performante. Le metriche di precisione, recall e F1 Score sono tutte molto alte, indicando un eccellente bilanciamento tra falsi positivi e falsi negativi. L'AUC-ROC quasi perfetto suggerisce che il modello ha un'ottima capacità discriminativa.

