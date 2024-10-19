

### [Parameters](./parameters.txt)

accuracy test set: 0.9860140085220337
accuracy train set: 0.9984992742538452
accuracy validation set: 0.9929947257041931
loss test set: 0.07135152071714401
loss train set: 0.010861841030418873
loss validation set: 0.053048450499773026

F1 Score: 0.9860452931283162
Precision: 0.9866395259751902
Recall: 0.986013986013986
AUC-ROC: 0.9993158761762544

1. Analisi dei Risultati

### Accuratezza
- **Test Set**: 0.9860
- **Train Set**: 0.9985
- **Validation Set**: 0.9930

L'accuratezza misura la percentuale di previsioni corrette. I valori mostrano che il modello ha un'alta accuratezza su tutti i set, con una leggera differenza tra il set di addestramento e quelli di test e validazione. Questo suggerisce che il modello generalizza bene e non è sovra-addestrato.

### Perdita (Loss)
- **Test Set**: 0.0714
- **Train Set**: 0.0109
- **Validation Set**: 0.0530

La perdita quantifica quanto le previsioni del modello si discostano dai valori reali. Valori bassi indicano buone prestazioni. La perdita è molto bassa nel set di addestramento, leggermente più alta nei set di test e validazione, ma comunque bassa, indicando che il modello è ben addestrato.

### F1 Score
- **F1 Score**: 0.9860

L'F1 Score è la media armonica di precisione e richiamo, utile per valutare modelli su dati sbilanciati. Un valore di 0.9860 indica un buon equilibrio tra precisione e richiamo.

### Precisione e Richiamo
- **Precisione**: 0.9866
- **Richiamo**: 0.9860

- **Precisione**: La percentuale di veri positivi tra tutte le previsioni positive. Un valore di 0.9866 indica che il modello ha pochi falsi positivi.
- **Richiamo**: La percentuale di veri positivi tra tutti i casi effettivamente positivi. Un valore di 0.9860 indica che il modello ha pochi falsi negativi.

### AUC-ROC
- **AUC-ROC**: 0.9993

L'AUC-ROC misura la capacità del modello di distinguere tra classi. Un valore vicino a 1 indica eccellenti capacità di discriminazione.

### Conclusione
I risultati indicano che il modello è altamente performante, con ottima accuratezza, precisione, richiamo e capacità di discriminazione. La leggera differenza tra i set di addestramento e quelli di test/validazione suggerisce che il modello è ben generalizzato e non soffre di overfitting.

