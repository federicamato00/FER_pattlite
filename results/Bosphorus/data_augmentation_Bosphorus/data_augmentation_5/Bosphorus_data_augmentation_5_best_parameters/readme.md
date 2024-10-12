# Risultati e Grafici
 
Possiamo fare le seguenti osservazioni:

## Grafico 1: [confusion_matrix](./confusion_matrix.png)

### Interpretazione:
- **Righe e Colonne**: Le righe rappresentano le emozioni reali, mentre le colonne rappresentano le emozioni previste dal modello.
- **Diagonale Principale**: I numeri sulla diagonale (ad esempio, 43 per "rabbia" e 44 per "felicità") indicano le previsioni corrette. Più alto è il numero, migliore è la prestazione del modello per quella specifica emozione.
- **Errori di Previsione**: I numeri fuori dalla diagonale mostrano le previsioni errate. Ad esempio, il modello ha previsto "disgusto" come "rabbia" 1 volta.

### Esempio:
- **Rabbia**: 43 previsioni corrette, 1 errore come "disgusto".
- **Felicità**: 44 previsioni corrette, nessun errore.

Questa matrice aiuta a identificare dove il modello è più preciso e dove potrebbe migliorare.

## Grafico 2: [Training e Validation plots](./training_validation_plots.png)

Ci sono due grafici:

### Grafico a Sinistra: Accuratezza di Addestramento e Validazione
- **Linea Blu (Training Accuracy)**: Mostra l'accuratezza del modello sui dati di addestramento. Parte da 0 e sale fino a quasi 1 (100%), indicando che il modello migliora significativamente.
- **Linea Arancione (Validation Accuracy)**: Mostra l'accuratezza del modello sui dati di validazione. Segue un trend simile alla linea blu, ma con alcune fluttuazioni, suggerendo che il modello generalizza bene sui dati non visti.

### Grafico a Destra: Perdita di Addestramento e Validazione
- **Linea Blu (Training Loss)**: Mostra la perdita del modello sui dati di addestramento. Diminuisce rapidamente, indicando che il modello sta imparando.
- **Linea Arancione (Validation Loss)**: Mostra la perdita sui dati di validazione. Anche questa diminuisce, ma con più fluttuazioni, suggerendo che il modello potrebbe essere leggermente sovra-addestrato.

Questi grafici indicano che il modello sta migliorando sia in termini di accuratezza che di perdita, ma è importante monitorare le fluttuazioni nella perdita di validazione per evitare il sovra-addestramento. 

## [Parameters](./parameters.txt)
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./parameters.txt). Per questa prima prova, si sono ottenuti i seguenti risultati:

- accuracy test set: 94.5%
- accuracy train set: 95.27%
- accuracy validation set: 91.88%

Guardando questi nuovi risultati, possiamo fare alcune considerazioni:

1. **Accuratezza Complessiva**

- **Training Set (95.27%)**: Il modello ha un'alta accuratezza sui dati di addestramento, indicando che ha imparato bene dai dati forniti.
- **Validation Set (91.88%)**: L'accuratezza sui dati di validazione è inferiore rispetto a quella di addestramento, suggerendo che il modello potrebbe non generalizzare altrettanto bene sui dati non visti.
- **Test Set (94.5%)**: L'accuratezza sui dati di test è intermedia tra quella di addestramento e quella di validazione, indicando che il modello mantiene buone prestazioni su dati nuovi, ma non perfette.

2. **Possibile Overfitting**
- La differenza tra l'accuratezza di addestramento e quella di validazione (circa 3.39%) potrebbe indicare un leggero overfitting. Il modello potrebbe aver imparato troppo bene i dettagli specifici dei dati di addestramento, perdendo un po' di capacità di generalizzare.

3. **Generalizzazione Moderata**
- L'accuratezza di validazione più bassa rispetto a quella di addestramento suggerisce che il modello potrebbe avere difficoltà a generalizzare su dati non visti. Questo è un segnale che potrebbe essere necessario migliorare la capacità di generalizzazione del modello, magari utilizzando tecniche di regolarizzazione o aumentando la quantità di dati di addestramento.

4. **Consistenza delle Prestazioni**
- Nonostante la differenza tra addestramento e validazione, l'accuratezza del test set è ancora alta (94.5%), indicando che il modello è comunque abbastanza affidabile.

### Considerazioni Finali
Questi risultati suggeriscono che il modello è ben addestrato ma potrebbe beneficiare di ulteriori miglioramenti per ridurre l'overfitting e migliorare la generalizzazione. Potresti considerare tecniche come la regolarizzazione, il dropout, o l'aumento dei dati di addestramento per migliorare le prestazioni.


**Tutti i parametri usati sono presi da best_parameters.txt**

