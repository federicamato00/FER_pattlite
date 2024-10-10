# Risultati e Grafici
 
Possiamo fare le seguenti osservazioni:

## Grafico 1: [confusion_matrix](./confusion_matrix.png)

* **Diagionale principale ben popolata**: Il fatto che la diagonale principale della matrice sia ben popolata indica che il modello sta correttamente classificando la maggior parte delle immagini. Ciò suggerisce che il modello è in grado di distinguere in modo affidabile le diverse emozioni.
* **Bassa confusione tra alcune emozioni**: La bassa incidenza di errori di classificazione tra alcune coppie di emozioni (ad esempio, "anger" e "neutral") suggerisce che il modello è in grado di catturare le differenze sottili tra queste emozioni.


## Grafico 2: [Training e Validation plots](./training_validation_plots.png)

Questi grafici sono tipici dell'**apprendimento di modelli di machine learning**, in particolare di modelli di **classificazione**. Essi mostrano come il modello evolve nel tempo durante il processo di addestramento (training) e come si comporta su dati che non ha mai visto prima (validation).

Ci sono due grafici:

1. **Accuracy:**
    * **Training Accuracy:** Aumenta costantemente durante l'addestramento, indicando che il modello sta imparando sempre meglio a classificare correttamente i dati che ha già visto.
    * **Validation Accuracy:** Inizialmente aumenta, ma poi si stabilizza o addirittura diminuisce leggermente. Questo è un segnale che il modello sta iniziando a **overfittare** i dati di training, ovvero sta imparando a memoria le caratteristiche specifiche del training set, senza generalizzare bene a nuovi dati.

2. **Loss:**
    * **Training Loss:** Diminuisce costantemente, indicando che il modello sta diventando sempre più preciso nelle sue predizioni sui dati di training.
    * **Validation Loss:** Inizialmente diminuisce, ma poi raggiunge un minimo e inizia ad aumentare. Questo conferma il fenomeno dell'overfitting, ovvero il modello sta diventando troppo complesso e si adatta troppo ai "rumori" dei dati di training, perdendo la capacità di generalizzare.

**Cosa significano questi risultati:**

* **Il modello ha imparato a classificare i dati di training con una buona accuratezza.**
* **Il modello sta iniziando a overfittare i dati di training.**
* **L'addestramento potrebbe essere interrotto prima delle 200 epoche** per evitare un ulteriore deterioramento delle prestazioni su dati nuovi.

**Cosa si potrebbe fare per migliorare i risultati:**

* **Interrompere l'addestramento prima:** Si potrebbe usare una tecnica chiamata **early stopping** per interrompere l'addestramento quando la validation loss inizia ad aumentare.
* **Usare tecniche di regolarizzazione:** Come la L1 o L2 regularization, che penalizzano modelli troppo complessi e aiutano a prevenire l'overfitting.
* **Aumentare la dimensione del dataset:** Un dataset più grande e più diversificato può aiutare il modello a generalizzare meglio.
* **Modificare l'architettura del modello:** Si potrebbero provare modelli più semplici o più complessi, a seconda della natura dei dati.
* **Adattare gli hyperparametri:** Si potrebbero modificare parametri come il learning rate o il numero di neuroni per migliorare le prestazioni.

**In conclusione,** i grafici mostrano un modello che ha imparato bene sui dati di training, ma che potrebbe beneficiare di alcune modifiche per migliorare le sue prestazioni su dati nuovi.


## [Parameters](./parameters.txt)
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./parameters.txt) . Per questa prima prova, si sono ottenuti i seguenti risultati:

accuracy test set: 96.76%
accuracy train set: 96.47%
accuracy validation set: 95.13%

In questo caso è stato impostato il valore di batch_size a 16 e usato il modello di base di patt-lite.



