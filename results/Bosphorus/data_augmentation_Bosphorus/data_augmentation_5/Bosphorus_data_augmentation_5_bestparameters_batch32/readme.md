# Risultati e Grafici
 
Possiamo fare le seguenti osservazioni:

## Grafico 1: [confusion_matrix](./confusion_matrix.png)




## Grafico 2: [Training e Validation plots](./training_validation_plots.png)

Ci sono due grafici:


## [Parameters](./parameters.txt)
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./parameters.txt). Per questa prima prova, si sono ottenuti i seguenti risultati:


- accuracy test set: 94.17%
- accuracy train set: 95.83
- accuracy validation set: 94.97

Guardando questi risultati, possiamo fare alcune considerazioni:

1. **Alta Accuratezza Complessiva**

- **Training Set (95.82%)**: Il modello ha un'alta accuratezza sui dati di addestramento, indicando che ha imparato bene dai dati forniti.
- **Validation Set (94.97%)**: L'accuratezza sui dati di validazione è leggermente inferiore a quella di addestramento, ma comunque molto alta, suggerendo che il modello generalizza bene sui dati non visti.

- **Test Set (94.17%)**: L'accuratezza sui dati di test è molto vicina a quella di addestramento e validazione, confermando che il modello mantiene buone prestazioni anche su dati completamente nuovi.

2. **Assenza di Overfitting**
- La differenza minima tra l'accuratezza di addestramento e quella di validazione indica che il modello non è sovra-addestrato. In altre parole, non ha imparato solo i dettagli specifici dei dati di addestramento, ma ha anche generalizzato bene.

3. **Buona Generalizzazione**
- L'accuratezza di validazione leggermente inferiore a quella di addestramento potrebbe suggerire che il modello è ben bilanciato e non ha problemi di overfitting o underfitting.

4. **Consistenza delle Prestazioni**
- La coerenza tra le accuratezze di addestramento, validazione e test indica che il modello è stabile e affidabile.

### Considerazioni Finali
Questi risultati sono molto positivi e indicano che il modello è ben addestrato e generalizza bene sui dati non visti. Tuttavia, è sempre utile monitorare le prestazioni del modello su nuovi dati nel tempo per assicurarsi che continui a funzionare bene.

