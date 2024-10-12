# Risultati e Grafici
 
Possiamo fare le seguenti osservazioni:

## Grafico 1: [confusion_matrix](./confusion_matrix.png)

- **Righe e Colonne**: Le righe rappresentano le etichette reali (emozioni effettive), mentre le colonne rappresentano le etichette previste dal modello.
- **Valori sulla Diagonale**: I numeri sulla diagonale (ad esempio, 40 per "rabbia" e 43 per "felicità") indicano le previsioni corrette, dove l'emozione prevista corrisponde a quella reale.
- **Valori Fuori Diagonale**: I numeri fuori dalla diagonale (ad esempio, 1 per "rabbia" prevista come "disgusto") indicano errori di previsione.

### Interpretazione:
- **Prestazioni del Modello**: Un modello ideale avrebbe tutti i valori sulla diagonale, indicando previsioni perfette. I valori fuori dalla diagonale mostrano dove il modello ha sbagliato.
- **Esempio**: Il modello ha previsto correttamente "rabbia" 40 volte, ma ha confuso "rabbia" con "disgusto" 1 volta.



## Grafico 2: [Training e Validation plots](./training_validation_plots.png)

Ci sono due grafici:

### Grafico a Sinistra: Accuratezza di Addestramento e Validazione
- **Linea Blu (Training Accuracy)**: Mostra l'accuratezza del modello sui dati di addestramento. Parte da circa 0.5 e sale fino a circa 0.95, indicando che il modello migliora nel tempo.
- **Linea Arancione (Validation Accuracy)**: Mostra l'accuratezza del modello sui dati di validazione. Segue un trend simile alla linea blu, ma leggermente inferiore, indicando che il modello generalizza bene sui dati non visti.

### Grafico a Destra: Perdita di Addestramento e Validazione
- **Linea Blu (Training Loss)**: Mostra la perdita del modello sui dati di addestramento. Diminuisce rapidamente da circa 1.4 a quasi 0, indicando che il modello sta imparando.
- **Linea Arancione (Validation Loss)**: Mostra la perdita sui dati di validazione. Anche questa diminuisce, ma con più fluttuazioni, suggerendo che il modello potrebbe essere leggermente sovra-addestrato.

Questi grafici indicano che il modello sta migliorando sia in termini di accuratezza che di perdita, ma è importante monitorare le fluttuazioni nella perdita di validazione per evitare il sovra-addestramento. 

## [Parameters](./parameters.txt)
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./parameters.txt). Per questa prima prova, si sono ottenuti i seguenti risultati:

- accuracy test set: 92.56%
- accuracy train set: 94.15%
- accuracy validation set: 94.16%

Il batch size è in questo caso impostato a 8, si seguono i best_hyperparameters trovati.



