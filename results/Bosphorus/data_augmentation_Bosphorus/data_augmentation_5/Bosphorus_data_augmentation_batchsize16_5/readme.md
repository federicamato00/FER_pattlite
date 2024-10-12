# Risultati e Grafici
 
Possiamo fare le seguenti osservazioni:

## Grafico 1: [confusion_matrix](./confusion_matrix.png)

- **Classi**: Le classi rappresentate sono 'anger' (rabbia), 'disgust' (disgusto), 'fear' (paura), 'happiness' (felicità), 'sadness' (tristezza), 'surprise' (sorpresa) e 'neutral' (neutrale).
- **Predizioni Corrette**: I numeri sulla diagonale principale indicano le predizioni corrette per ciascuna emozione. Ad esempio, ci sono **44 predizioni corrette per 'anger'** e **44 per 'happiness'**.
- **Misclassificazioni**: I numeri fuori dalla diagonale principale rappresentano le misclassificazioni. Ad esempio, ci sono **6 casi in cui 'sadness' è stato classificato come 'anger'**.
- **Colori**: La gradazione di colore aiuta a visualizzare le aree di confusione: i colori più chiari indicano meno confusione, mentre i colori più scuri indicano una maggiore confusione tra le diverse emozioni.

### Significato
- **Accuratezza**: La matrice mostra che il modello ha un'alta accuratezza per alcune emozioni come 'anger' e 'happiness', ma potrebbe avere difficoltà a distinguere tra altre emozioni.
- **Aree di Miglioramento**: Le misclassificazioni indicano le aree in cui il modello potrebbe essere migliorato, ad esempio, riducendo la confusione tra 'sadness' e 'anger'.


## Grafico 2: [Training e Validation plots](./training_validation_plots.png)

Ci sono due grafici:
Questi due grafici mostrano l'andamento dell'accuratezza e della perdita durante l'addestramento e la validazione di un modello di machine learning.

- **Accuratezza**: L’accuratezza di addestramento è molto alta, quasi 1.0, e l’accuratezza di validazione è sopra 0.9. Questo indica che il modello sta imparando bene dai dati di addestramento e generalizza abbastanza bene sui dati di validazione.
- **Perdita**: La perdita di addestramento è molto bassa, vicino a zero, mentre la perdita di validazione è stabile poco sopra 0.2. Questo è un buon segno, ma la differenza tra le due perdite potrebbe indicare un leggero overfitting, dove il modello si adatta troppo ai dati di addestramento.
- **Convergenza**: Entrambi i grafici mostrano che dopo circa 50 epoche, i miglioramenti diventano meno significativi. Questo suggerisce che il modello ha raggiunto un punto di convergenza, dove ulteriori addestramenti non portano a grandi miglioramenti.

Questi grafici indicano che il modello sta imparando bene, ma dopo un certo punto, i miglioramenti diventano meno significativi.


## [Parameters](./parameters.txt)
I parametri usati per l'addestramento di questo modello, sono raccolti nel file denominato [parameters.txt](./parameters.txt). Per questa prima prova, si sono ottenuti i seguenti risultati:

- accuracy test set: 95.79%
- accuracy train set: 95.96%
- accuracy validation set: 96.59%

Il batch size è in questo caso impostato a 16, si seguono i best_hyperparameters trovati.

Guardando questi risultati, possiamo fare alcune considerazioni importanti:

1. **Alta Accuratezza Complessiva**
- **Training Set (95.96%)**: Il modello ha un'alta accuratezza sui dati di addestramento, indicando che ha imparato bene dai dati forniti.
- **Validation Set (96.59%)**: L'accuratezza sui dati di validazione è leggermente superiore a quella sui dati di addestramento, suggerendo che il modello generalizza bene sui dati non visti.
- **Test Set (95.79%)**: L'accuratezza sui dati di test è molto vicina a quella di addestramento e validazione, confermando che il modello mantiene buone prestazioni anche su dati completamente nuovi.

2. **Assenza di Overfitting**
- La differenza minima tra l'accuratezza di addestramento e quella di validazione indica che il modello non è sovra-addestrato. In altre parole, non ha imparato solo i dettagli specifici dei dati di addestramento, ma ha anche generalizzato bene.

3. **Buona Generalizzazione**
- L'accuratezza di validazione leggermente superiore a quella di addestramento potrebbe suggerire che il modello è ben bilanciato e non ha problemi di overfitting o underfitting.

4. **Consistenza delle Prestazioni**
- La coerenza tra le accuratezze di addestramento, validazione e test indica che il modello è stabile e affidabile.

Considerazioni Finali
Questi risultati sono molto positivi e indicano che il modello è ben addestrato e generalizza bene sui dati non visti. Tuttavia, è sempre utile monitorare le prestazioni del modello su nuovi dati nel tempo per assicurarsi che continui a funzionare bene.

