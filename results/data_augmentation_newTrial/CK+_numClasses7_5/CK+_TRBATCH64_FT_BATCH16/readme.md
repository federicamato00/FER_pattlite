
### [Parameters](./parameters.txt)

accuracy test set: 0.9755244851112366
accuracy train set: 0.9959980249404907
accuracy validation set: 0.996497392654419
loss test set: 0.09093529731035233
loss train set: 0.02337195724248886
loss validation set: 0.037260428071022034

F1 Score: 0.9753897327355717
Precision: 0.9755810900488897
Recall: 0.9755244755244755
AUC-ROC: 0.9995567565005096

1. Analisi dei Risultati

### Accuratezza
- **Test Set**: 0.9755
- **Train Set**: 0.9960
- **Validation Set**: 0.9965

L'accuratezza misura la proporzione di previsioni corrette. I valori sono molto alti per tutti i set, indicando che il modello è molto preciso sia sui dati di addestramento che su quelli di test e validazione.

### Perdita (Loss)
- **Test Set**: 0.0909
- **Train Set**: 0.0234
- **Validation Set**: 0.0373

La perdita misura quanto le previsioni del modello si discostano dai valori reali. Valori più bassi indicano una migliore performance. La perdita è molto bassa per il set di addestramento, leggermente più alta per il set di validazione e test, ma comunque bassa, suggerendo che il modello generalizza bene.

### F1 Score
- **F1 Score**: 0.9754

L'F1 Score è la media armonica di precisione e recall. Un valore di 0.9754 indica un buon equilibrio tra precisione e recall.

### Precisione
- **Precisione**: 0.9756

La precisione misura la proporzione di veri positivi tra tutti i positivi predetti. Un valore di 0.9756 indica che il modello ha pochi falsi positivi.

### Recall
- **Recall**: 0.9755

Il recall misura la proporzione di veri positivi tra tutti i positivi reali. Un valore di 0.9755 indica che il modello ha pochi falsi negativi.

### AUC-ROC
- **AUC-ROC**: 0.9996

L'AUC-ROC misura la capacità del modello di distinguere tra classi. Un valore vicino a 1 indica un'eccellente capacità discriminativa.

### Conclusione
I risultati indicano che il modello è molto performante, con alta accuratezza, bassa perdita e ottimi valori di precisione, recall e F1 Score. L'AUC-ROC quasi perfetto suggerisce che il modello è molto efficace nel distinguere tra le classi.

