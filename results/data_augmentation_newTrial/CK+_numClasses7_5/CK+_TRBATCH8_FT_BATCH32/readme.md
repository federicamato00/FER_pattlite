### [Parameters](./parameters.txt)

accuracy test set: 0.9790209531784058
accuracy train set: 0.9974987506866455
accuracy validation set: 0.9912434220314026
loss test set: 0.06381405889987946
loss train set: 0.01248881034553051
loss validation set: 0.03707266226410866

F1 Score: 0.9791431590126369
Precision: 0.9799193055007008
Recall: 0.9790209790209791
AUC-ROC: 0.9998571902628647

1. Analisi Risultati

    1. **Accuratezza**: 
    - **Train Set**: Un'accuratezza molto alta (0.9975) indica che il modello sta imparando molto bene dai dati di addestramento.
    - **Validation Set**: Un'accuratezza di 0.9912 è eccellente e suggerisce che il modello generalizza bene sui dati non visti.
    - **Test Set**: Un'accuratezza di 0.9790 è ancora molto alta, confermando che il modello mantiene buone performance anche sui dati di test.

    2. **Loss**:
    - **Train Set**: Una loss molto bassa (0.0125) indica che il modello sta minimizzando l'errore sui dati di addestramento.
    - **Validation Set**: Una loss di 0.0371 è bassa e suggerisce che il modello non sta overfittando eccessivamente.
    - **Test Set**: Una loss di 0.0638 è accettabile e conferma che il modello mantiene buone performance sui dati di test.

    3. **F1 Score, Precision, Recall**:
    - **F1 Score**: Un valore di 0.9791 indica un buon equilibrio tra precisione e recall.
    - **Precision**: Una precisione di 0.9799 significa che il 97.99% delle predizioni positive sono corrette.
    - **Recall**: Un recall di 0.9790 indica che il modello è in grado di identificare il 97.90% delle istanze positive.

    4. **AUC-ROC**: 
    - Un valore di 0.9999 è eccellente e suggerisce che il modello ha una capacità quasi perfetta di distinguere tra le classi positive e negative.

    5. Considerazioni
    - **Bilanciamento**: Le metriche di precisione e recall sono bilanciate, il che è un buon segno che il modello non sta sacrificando una metrica per migliorare l'altra.
    - **Performance**: L'alto valore dell'AUC-ROC indica che il modello è molto robusto e performante.

