

### [Parameters](./parameters.txt)

accuracy test set: 0.9708737730979919
accuracy train set: 0.9953595995903015
accuracy validation set: 0.9788960814476013
loss test set: 0.08992747217416763
loss train set: 0.03127969428896904
loss validation set: 0.10491835325956345

F1 Score: 0.9706933952170618
Precision: 0.9716626761736873
Recall: 0.970873786407767
AUC-ROC: 0.9996002250719231


1. Analisi dei risultati

1. Accuratezza

    - **Test Set**: 0.9709
    - **Train Set**: 0.9954
    - **Validation Set**: 0.9789

    L'accuratezza è alta in tutti i set, il che indica che il modello sta performando bene sia sui dati di addestramento che sui dati di test e di validazione. Tuttavia, l'accuratezza del set di addestramento è leggermente più alta, il che potrebbe suggerire un leggero overfitting.

2. Loss
    - **Test Set**: 0.0899
    - **Train Set**: 0.0313
    - **Validation Set**: 0.1049

    Le perdite sono basse, il che è un buon segno. La perdita di addestramento è significativamente più bassa rispetto a quella di test e di validazione, il che potrebbe confermare il sospetto di overfitting.

3. Metriche di Valutazione
    - **F1 Score**: 0.9707
    - **Precision**: 0.9717
    - **Recall**: 0.9709
    - **AUC-ROC**: 0.9996

    Queste metriche indicano che il modello ha un ottimo bilanciamento tra precisione e richiamo, con un F1 Score molto alto. L'AUC-ROC vicino a 1 suggerisce che il modello è eccellente nel distinguere tra le classi.

3. Considerazioni Finali
    - **Overfitting**: La differenza tra l'accuratezza e la perdita del set di addestramento rispetto ai set di test e di validazione potrebbe indicare un leggero overfitting. Potresti considerare tecniche come la regolarizzazione o l'uso di più dati di addestramento per mitigare questo problema.
    - **Performance Generale**: Nel complesso, il modello sembra performare molto bene con metriche di valutazione eccellenti.

