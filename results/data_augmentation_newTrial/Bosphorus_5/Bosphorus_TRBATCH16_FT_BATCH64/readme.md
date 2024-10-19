

### [Parameters](./parameters.txt)

accuracy test set: 0.9902912378311157
accuracy train set: 0.998607873916626
accuracy validation set: 0.9788960814476013
loss test set: 0.05602969974279404
loss train set: 0.018709290772676468
loss validation set: 0.10876935720443726

F1 Score: 0.9902870996353607
Precision: 0.9906430280005627
Recall: 0.9902912621359223
AUC-ROC: 0.9997306851080436


1. Accuratezza
- **Test Set**: 0.9903
- **Train Set**: 0.9986
- **Validation Set**: 0.9789

L'accuratezza è estremamente alta in tutti i set, suggerendo che il modello è molto preciso nelle sue previsioni.

2. Loss
- **Test Set**: 0.0560
- **Train Set**: 0.0187
- **Validation Set**: 0.1088

La loss è molto bassa, specialmente nel train set, il che indica che il modello ha appreso molto bene i dati di addestramento. La differenza tra la loss del train set e quella del validation set è ridotta, suggerendo un miglioramento nella generalizzazione del modello.

3. Metriche di Valutazione
- **F1 Score**: 0.9903
- **Precision**: 0.9906
- **Recall**: 0.9903
- **AUC-ROC**: 0.9997

Queste metriche sono eccellenti:
- **F1 Score**: Indica un ottimo equilibrio tra precisione e recall.
- **Precision**: Alta precisione significa che il modello ha pochissimi falsi positivi.
- **Recall**: Alto recall indica che il modello ha pochissimi falsi negativi.
- **AUC-ROC**: Un valore vicino a 1 indica che il modello è estremamente efficace nel distinguere tra le classi.

4. Considerazioni Finali
Questi risultati mostrano un modello altamente performante e ben bilanciato. L'accuratezza e le metriche di valutazione suggeriscono che il modello è molto robusto e generalizza bene sui dati non visti. La riduzione della loss nel validation set rispetto ai risultati precedenti è un segnale positivo.
