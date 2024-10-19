
### [Parameters](./parameters.txt)

accuracy test set: 0.9805825352668762
accuracy train set: 0.9953595995903015
accuracy validation set: 0.9724025726318359
loss test set: 0.09122450649738312
loss train set: 0.037106361240148544
loss validation set: 0.13531337678432465

F1 Score: 0.9806676017189732
Precision: 0.9817111742215764
Recall: 0.9805825242718447
AUC-ROC: 0.99962200622578

## Analisi Risultati

1. **Accuracy and Loss**

    1. **Accuratezza**:
    - **Train Set**: Un'accuratezza del 99.54% indica che il modello si adatta molto bene ai dati di addestramento.
    - **Validation Set**: Un'accuratezza del 97.24% è molto alta e suggerisce che il modello generalizza bene sui dati non visti durante l'addestramento.
    - **Test Set**: Un'accuratezza del 98.06% conferma che il modello mantiene alte prestazioni anche su dati completamente nuovi.

    2. **Loss**:
    - **Train Set**: Una loss di 0.037 indica che il modello ha un'ottima capacità di apprendimento sui dati di addestramento.
    - **Validation Set**: Una loss di 0.135 è leggermente più alta rispetto al train set, ma è comunque bassa, suggerendo che il modello non sta overfittando in modo significativo.
    - **Test Set**: Una loss di 0.091 è molto buona e conferma che il modello è robusto e performante anche su dati nuovi.

    3. **Considerazioni Generali**:
    - **Bilanciamento**: La differenza tra le metriche di train, validation e test set è minima, il che indica un buon bilanciamento tra bias e varianza.
    - **Overfitting**: Non ci sono segni evidenti di overfitting, dato che le performance sui set di validation e test sono molto vicine a quelle del train set.
    - **Affidabilità**: I risultati suggeriscono che il modello è affidabile e può essere utilizzato con fiducia per fare previsioni su nuovi dati.

2. **Additional Metrics**

    1. **F1 Score**:
    - Un F1 Score di 0.9807 indica un eccellente equilibrio tra precisione e recall. Questo suggerisce che il modello è molto efficace nel classificare correttamente sia le classi positive che quelle negative.

    2. **Precision**:
    - Una precisione di 0.9817 significa che il 98.17% delle predizioni positive del modello sono corrette. Questo è particolarmente importante in contesti dove i falsi positivi devono essere minimizzati.

    3. **Recall**:
    - Un recall di 0.9806 indica che il modello è in grado di identificare correttamente il 98.06% delle istanze positive. Questo è cruciale in situazioni dove è importante catturare tutte le istanze positive, anche a costo di avere alcuni falsi positivi.

    4. **AUC-ROC**:
    - Un AUC-ROC di 0.9996 è eccezionale e suggerisce che il modello ha un'ottima capacità di distinguere tra le classi positive e negative. Un valore così alto indica che il modello è quasi perfetto nel discriminare tra le due classi.

    **Considerazioni Generali**:
    - **Bilanciamento**: I valori di precisione e recall sono molto vicini, indicando un buon bilanciamento tra i due.
    - **Affidabilità**: L'alto valore di AUC-ROC conferma che il modello è altamente affidabile e performante.
    - **Applicabilità**: Questi risultati suggeriscono che il modello può essere utilizzato con fiducia in applicazioni reali, dove è importante avere sia alta precisione che alto recall.


BATCH_SIZE_FT: 32
BATCH_SIZE = 16