

accuracy test set: 0.9805825352668762
accuracy train set: 0.9902552366256714
accuracy validation set: 0.9788960814476013
loss test set: 0.11453861743211746
loss train set: 0.03563930094242096
loss validation set: 0.1584118753671646

F1 Score: 0.9806454892019132
Precision: 0.9812018132111772
Recall: 0.9805825242718447
AUC-ROC: 0.9995999981849037

## Analisi Risultati

### Accuratezza
    - **Train set**: 0.9903
    - **Validation set**: 0.9789
    - **Test set**: 0.9806

### Loss:
    - **Train set**: 0.0356
    - **Validation set**: 0.1584
    - **Test set**: 0.1145

1. **Accuracy e Loss**

    1. **Accuratezza**:
    - **Train set**: Un'accuratezza del 99.03% indica che il modello si comporta molto bene sui dati di addestramento.
    - **Validation set**: Un'accuratezza del 97.89% sui dati di validazione è molto buona e suggerisce che il modello generalizza bene.
    - **Test set**: Un'accuratezza del 98.06% sui dati di test conferma che il modello mantiene buone prestazioni anche su dati non visti.

    2. **Loss**:
    - **Train set**: Un valore di loss molto basso (0.0356) sui dati di addestramento indica che il modello ha appreso bene i pattern nei dati.
    - **Validation set**: Un valore di loss più alto (0.1584) rispetto al train set può indicare una leggera overfitting, ma non è preoccupante dato che l'accuratezza è ancora alta.
    - **Test set**: Un valore di loss di 0.1145 sui dati di test è accettabile e conferma che il modello non ha problemi significativi di overfitting.


2. **Additional metrics** 
    - **F1 Score**: 0.9806
    - **Precision**: 0.9812
    - **Recall**: 0.9806
    - **AUC-ROC**: 0.9996

    1. **F1 Score**:
    - Un F1 Score di 0.9806 indica un buon equilibrio tra precisione e recall. È una metrica utile quando hai bisogno di bilanciare entrambi gli aspetti, specialmente in presenza di classi sbilanciate.

    2. **Precision**:
    - Una precisione di 0.9812 significa che il 98.12% delle previsioni positive del modello sono corrette. Questo è importante quando il costo di un falso positivo è alto.

    3. **Recall**:
    - Un recall di 0.9806 indica che il modello è in grado di identificare correttamente il 98.06% delle istanze positive. Questo è cruciale quando il costo di un falso negativo è alto.

    4. **AUC-ROC**:
    - Un valore di AUC-ROC di 0.9996 è eccellente e suggerisce che il modello è molto efficace nel distinguere tra le classi positive e negative. Un valore vicino a 1 indica un modello quasi perfetto.

3. **Conclusione**
    Questi risultati confermano ulteriormente che il modello è molto performante. Le metriche di precisione, recall e F1 Score sono tutte molto alte, indicando un buon bilanciamento tra falsi positivi e falsi negativi. L'AUC-ROC quasi perfetto suggerisce che il modello ha un'ottima capacità discriminativa.



