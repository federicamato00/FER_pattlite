
# Risultati e Grafici
 
Possiamo fare le seguenti osservazioni:

## Grafico 1: [confusion_matrix](./confusion_matrix.png)




## Grafico 2: [Training e Validation plots](./training_validation_plots.png)

Ci sono due grafici:


## [Parameters](./parameters.txt)
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./parameters.txt). Per questa prima prova, si sono ottenuti i seguenti risultati:

- accuracy test set: 97.73%
- accuracy train set: 98.56 %
- accuracy validation set: 97.72%

1. **Accuracy del Train Set: 0.9856**
- **Interpretazione**: Il modello ha un'accuratezza molto alta sui dati di addestramento, il che significa che ha imparato bene i pattern presenti nei dati.
- **Possibili Considerazioni**: Un'accuratezza così alta potrebbe indicare che il modello è molto complesso e ha una buona capacità di adattarsi ai dati di addestramento. Tuttavia, è importante verificare che non ci sia overfitting.

2. **Accuracy del Validation Set: 0.9773**
- **Interpretazione**: L'accuratezza sui dati di validazione è leggermente inferiore a quella sui dati di addestramento, ma molto vicina. Questo è un buon segno, poiché indica che il modello generalizza bene ai dati non visti durante l'addestramento.
- **Possibili Considerazioni**: La vicinanza tra l'accuratezza del train set e quella del validation set suggerisce che il modello non sta overfittando. Se ci fosse una grande differenza, potrebbe essere necessario rivedere il modello o i dati di addestramento.

3. **Accuracy del Test Set: 0.9773**
- **Interpretazione**: L'accuratezza sui dati di test è praticamente identica a quella sui dati di validazione. Questo conferma che il modello mantiene la sua performance anche su dati completamente nuovi.
- **Possibili Considerazioni**: Questo risultato è molto positivo, poiché indica che il modello è robusto e affidabile. La consistenza tra i set di dati suggerisce che il modello è ben bilanciato.

4. Conclusioni Generali
- **Buona Generalizzazione**: Il modello non solo si adatta bene ai dati di addestramento, ma generalizza anche bene ai dati di validazione e test.
- **Assenza di Overfitting**: La somiglianza tra le accuratezze dei diversi set di dati indica che non c'è un problema significativo di overfitting.
- **Affidabilità**: La consistenza delle performance su tutti i set di dati suggerisce che il modello è affidabile e può essere utilizzato con fiducia su nuovi dati.

