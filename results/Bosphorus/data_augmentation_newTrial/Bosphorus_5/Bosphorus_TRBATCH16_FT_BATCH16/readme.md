


## Analisi Risultati


1. **Accuracy and Loss**
    1. **Accuratezza**:
    - **Train Set**: Un'accuratezza del 99.35% è eccellente e mostra che il modello si adatta molto bene ai dati di addestramento.
    - **Validation Set**: Un'accuratezza del 98.21% è molto alta e indica che il modello generalizza bene sui dati di validazione.
    - **Test Set**: Un'accuratezza del 98.06% conferma che il modello mantiene alte prestazioni anche su dati completamente nuovi.

    2. **Loss**:
    - **Train Set**: Una loss di 0.026 è molto bassa, suggerendo che il modello ha appreso molto bene dai dati di addestramento.
    - **Validation Set**: Una loss di 0.079 è leggermente più alta rispetto al train set, ma è comunque bassa, indicando che il modello non sta overfittando in modo significativo.
    - **Test Set**: Una loss di 0.103 è ancora molto buona e conferma che il modello è robusto e performante anche su dati nuovi.

    3. **Considerazioni Generali**:
    - **Bilanciamento**: La differenza tra le metriche di train, validation e test set è minima, il che indica un buon bilanciamento tra bias e varianza.
    - **Overfitting**: Non ci sono segni evidenti di overfitting, dato che le performance sui set di validation e test sono molto vicine a quelle del train set.
    - **Affidabilità**: I risultati suggeriscono che il modello è affidabile e può essere utilizzato con fiducia per fare previsioni su nuovi dati.

2. **Additional Metrics**


    1. **F1 Score**: 
    - Un F1 Score di 0.9806 indica un eccellente equilibrio tra precisione e recall. Questo suggerisce che il modello è molto efficace nel classificare correttamente sia le classi positive che quelle negative.

    2. **Precision**:
    - Una precisione di 0.9817 significa che il 98.17% delle predizioni positive del modello sono corrette. Questo è particolarmente importante in contesti dove i falsi positivi devono essere minimizzati.

    3. **Recall**:
    - Un recall di 0.9806 indica che il modello è in grado di identificare correttamente il 98.06% delle istanze positive. Questo è cruciale in situazioni dove è importante catturare tutte le istanze positive, anche a costo di avere alcuni falsi positivi.

    4. **AUC-ROC**:
    - Un AUC-ROC di 0.9998 è eccezionale e suggerisce che il modello ha un'ottima capacità di distinguere tra le classi positive e negative. Un valore così alto indica che il modello è quasi perfetto nel discriminare tra le due classi.

    **Considerazioni Generali**:
    - **Bilanciamento**: I valori di precisione e recall sono molto vicini, indicando un buon bilanciamento tra i due.
    - **Affidabilità**: L'alto valore di AUC-ROC conferma che il modello è altamente affidabile e performante.
    - **Applicabilità**: Questi risultati suggeriscono che il modello può essere utilizzato con fiducia in applicazioni reali, dove è importante avere sia alta precisione che alto recall.


I risultati fanno riferimento ad un BATCH_SIZE = 16 e ad un BATCH_SIZE_FT = 16

