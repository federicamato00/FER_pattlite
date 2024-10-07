# Descrizione dei Grafici

Questa cartella contiene una serie di grafici che rappresentano vari dati e analisi. Di seguito è riportata una spiegazione dettagliata di ciascun grafico:

## Grafico 1: Distribuzioni delle classi prima dell'uso di SMOTE
![Grafico 1](/Users/federicaamato/Desktop/FER_pattlite/dataset_distribution/Bosphorus/prima_SMOTE.png)
Questo grafico mostra l'andamento della distribuzione dei dati in Bosphorus prima dell'uso di SMOTE per bilanciare la classi sbilanciate. Si può notare come l'emozione neutrale abbia più dati rispetto alle altre classi, questo potrebbe portare ad un problema di classificazione della  rete neurale che potrebbe infatti non apprendere bene le caratteristiche relative alle altre emozioni

## Grafico 2: Distribuzioni delle classi dopo l'uso di SMOTE
![Grafico 2](/Users/federicaamato/Desktop/FER_pattlite/dataset_distribution/Bosphorus/dopo_SMOTE.png)
Questo grafico illustra l'andamento della distribuzione dei dati in Bosphorus dopo l'uso di SMOTE per bilanciare la classi sbilanciate. Si può notare come ora ogni emozione abbia una distribuzione uguale.

## Grafico 3: Scatter Plot prima di SMOTE
![Grafico 3](/Users/federicaamato/Desktop/FER_pattlite/dataset_distribution/Bosphorus/scatter_prima_SMOTE.png)
Questo grafico mostra la distribuzione delle classi.
Dal grafico si possono fare alcune considerazioni sulle classi di emozioni rappresentate:

1. **Separabilità delle Classi**: Le diverse emozioni (rabbia, disgusto, paura, felicità, tristezza, sorpresa, neutrale) sono rappresentate da punti di colori diversi. La distribuzione dei punti mostra che alcune emozioni sono più facilmente separabili dalle altre, mentre altre hanno una maggiore sovrapposizione. Ad esempio, se i punti di rabbia e disgusto sono molto vicini, potrebbe essere difficile distinguere tra queste due emozioni.

2. **Distribuzione e Densità**: La densità dei punti per ciascuna emozione può indicare la frequenza di quella particolare emozione nel dataset. Se un'emozione ha molti punti concentrati in una zona, significa che è più comune rispetto ad altre emozioni che hanno meno punti.

3. **Effetto della Riduzione Dimensionale**: L'uso delle componenti principali (PCA) per ridurre la dimensionalità dei dati aiuta a visualizzare come le emozioni si distribuiscono nello spazio a due dimensioni. Questo può dare un'idea di come le emozioni si raggruppano e quanto sono distinte l'una dall'altra.

4. **Necessità di Bilanciamento**: La presenza di cluster molto densi per alcune emozioni e meno densi per altre suggerisce che il dataset potrebbe essere sbilanciato. Questo è il motivo per cui si sta considerando l'uso di SMOTE per bilanciare le classi.

Queste osservazioni possono aiutare a capire meglio la struttura del dataset e a prendere decisioni informate su come procedere con l'analisi e il bilanciamento dei dati. 

## Grafico 4: Scatter Plot dopo di SMOTE
![Grafico 4](/Users/federicaamato/Desktop/FER_pattlite/dataset_distribution/Bosphorus/scatter_dopo_SMOTE.png)
Questo scatter plot mostra la **distribuzione dei dati dopo l'applicazione di SMOTE**. Alcune considerazioni:

1. **Bilanciamento delle Classi**: Dopo l'applicazione di SMOTE, le classi sembrano più bilanciate, con una distribuzione più uniforme dei punti per ogni emozione.

2. **Separabilità delle Emozioni**: Nonostante il bilanciamento, c'è ancora una significativa sovrapposizione tra le diverse emozioni. Questo potrebbe rendere difficile per una rete neurale distinguere accuratamente tra alcune emozioni.

3. **Densità dei Punti**: La densità dei punti vicino all'origine suggerisce che molte emozioni condividono caratteristiche simili, complicando ulteriormente la classificazione.

4. **Utilizzo di PCA**: L'uso delle componenti principali (PCA) per ridurre la dimensionalità aiuta a visualizzare i dati, ma potrebbe non catturare tutte le complessità necessarie per una classificazione accurata.

In sintesi, mentre il bilanciamento dei dati è migliorato, la sovrapposizione tra le classi potrebbe rappresentare una sfida per una rete neurale che deve classificare le emozioni. Potrebbero essere necessarie ulteriori tecniche di pre-elaborazione o modelli più complessi per migliorare la precisione della classificazione.

## Grafico 5: Scatter Plot dopo di LDA 
![Grafico 4](/Users/federicaamato/Desktop/FER_pattlite/dataset_distribution/Bosphorus/scatter_dopo_LDA.png)

Questo grafico mostra lo scatter plot dei dati dopo aver usato LDA per mantenere le caratteristiche più discriminatorie del dataset. Si può vedere come la sovrapposizione tra le classi è migliorata, per cui ora potrebbe essere più semplice per il modello classificare il maniera efficiente le emozioni.

## Conclusioni

