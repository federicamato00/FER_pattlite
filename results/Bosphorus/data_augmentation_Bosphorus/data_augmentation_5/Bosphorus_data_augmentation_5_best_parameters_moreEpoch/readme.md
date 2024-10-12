

# Risultati e Grafici
 
Possiamo fare le seguenti osservazioni:

## Grafico 1: [confusion_matrix](./confusion_matrix.png)




## Grafico 2: [Training e Validation plots](./training_validation_plots.png)

Ci sono due grafici:


## [Parameters](./parameters.txt)
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./parameters.txt). Per questa prima prova, si sono ottenuti i seguenti risultati:


accuracy test set: 96.76%
accuracy train set: 97.54%
accuracy validation set: 96.92%


Guardando questi risultati, possiamo fare alcune considerazioni importanti:

1. **Alta Accuratezza Complessiva**
- **Training Set (97.54%)**: Il modello ha un'alta accuratezza sui dati di addestramento, indicando che ha imparato molto bene dai dati forniti.
- **Validation Set (96.92%)**: L'accuratezza sui dati di validazione è molto vicina a quella di addestramento, suggerendo che il modello generalizza bene sui dati non visti.
- **Test Set (96.76%)**: L'accuratezza sui dati di test è anch'essa molto alta e vicina a quelle di addestramento e validazione, confermando che il modello mantiene buone prestazioni anche su dati completamente nuovi.

2. **Assenza di Overfitting**
- La differenza minima tra l'accuratezza di addestramento e quella di validazione (circa 0.62%) indica che il modello non è sovra-addestrato. Questo significa che il modello non ha imparato solo i dettagli specifici dei dati di addestramento, ma ha anche generalizzato bene.

3. **Buona Generalizzazione**
- L'accuratezza di validazione molto vicina a quella di addestramento suggerisce che il modello è ben bilanciato e non ha problemi di overfitting o underfitting.

4. **Consistenza delle Prestazioni**
- La coerenza tra le accuratezze di addestramento, validazione e test indica che il modello è stabile e affidabile. Questo è un segnale molto positivo, poiché mostra che il modello è in grado di mantenere le sue prestazioni su diversi set di dati.

Considerazioni Finali
Questi risultati sono eccellenti e indicano che il modello è ben addestrato e generalizza molto bene sui dati non visti. Tuttavia, è sempre utile monitorare le prestazioni del modello su nuovi dati nel tempo per assicurarsi che continui a funzionare bene.

