# Risultati dei Codici

In questo documento, vengono commentati tutti i risultati ottenuti dai codici eseguiti.

## Indice
1. [Introduzione](#introduzione)
2. [Modifiche](#modifiche)
3. [Conclusioni](#conclusioni)

## Introduzione
In questa sezione, viene fornita una panoramica generale degli obiettivi e delle metodologie utilizzate nei codici.
Questo modello di FER_pattlite è stato dapprima valutato seguendo il modello di base fornito dal repository ufficiale.
I risultati per questa parte di analisi sono presenti nella cartella [BASE_MODEL](./BASE_MODEL/), in cui sono presenti i modelli base per il dataset [Bosphorus](./BASE_MODEL/Bosphorus/) e per il dataset [CK+](./BASE_MODEL/CK+/).

## Modifiche 
Per il dataset Bosphorus, si è ottenuto il seguente grafico per [accuracy e loss](./BASE_MODEL/Bosphorus/training_validation_plots.png), in cui si può notare come:

1. Grafico di Accuratezza
    - **Training Accuracy** (linea blu): Mostra come l'accuratezza del modello migliora durante l'addestramento. La linea tende a salire, indicando che il modello sta imparando bene dai dati di addestramento.
    - **Validation Accuracy** (linea arancione): Rappresenta l'accuratezza del modello sui dati di validazione. Anche questa linea sale, ma è generalmente inferiore alla linea blu, suggerendo che il modello potrebbe non generalizzare perfettamente ai nuovi dati.

2. Grafico di Loss
    - **Training Loss** (linea blu): Mostra la perdita del modello durante l'addestramento. La linea scende costantemente, indicando che il modello sta riducendo l'errore sui dati di addestramento.
    - **Validation Loss** (linea arancione): Rappresenta la perdita sui dati di validazione. Anche questa linea scende, ma con alcune fluttuazioni, suggerendo che il modello potrebbe avere difficoltà a generalizzare.

In sintesi, il modello sta migliorando sia in accuratezza che in perdita, ma c'è una leggera discrepanza tra i dati di addestramento e quelli di validazione, che potrebbe indicare un inizio di overfitting.

Per il dataset CK+, si è ottenuto il seguente grafico per [accuracy e loss](./BASE_MODEL/CK+/training_validation_plots.png), in cui si può notare come:

1. Grafico di Accuratezza
    - **Training Accuracy** (linea blu): Mostra un miglioramento generale con fluttuazioni significative. Questo indica che il modello sta imparando, ma con qualche instabilità.
    - **Validation Accuracy** (linea arancione): Rimane relativamente stabile con leggere fluttuazioni, suggerendo che il modello mantiene una performance costante sui dati di validazione.

2. Grafico di Loss
    - **Training Loss** (linea blu): Diminuisce nel tempo con alcune fluttuazioni, indicando che il modello sta riducendo l'errore sui dati di addestramento.
    - **Validation Loss** (linea arancione): Diminuisce inizialmente ma poi fluttua senza una chiara tendenza, suggerendo che il modello potrebbe avere difficoltà a generalizzare.

In sintesi, il modello mostra miglioramenti ma con alcune instabilità, specialmente nella perdita di validazione.

Alla luce di questi risultati, si è proceduto ad effettuare diverse prove per valutare come migliorare il modello e i relativi risultati.
Per prima cosa, si è proceduto a modificare il codice originale apportando le seguenti modifiche:

1. Ricerca dei migliori parametri: si è scritto il codice di ricerca dei migliori [hyperparameters](../patt-lite/hyperparameters_prova.py) in maniera tale da valutare quali potessero essere i migliori parametri da utilizzare nell'addestramento del modello e migliorare quindi le prestazioni finali del modello. La ricerca si divide nel seguente modo: 

    - Caricamento dei dati: I dati vengono caricati da un file HDF5 e suddivisi in set di addestramento, validazione e test. I dati vengono anche mescolati per garantire una distribuzione casuale.

    - Definizione del modello di addestramento: Viene definita una classe PattLiteHyperModel che utilizza Keras Tuner per cercare i migliori iperparametri per il modello di addestramento. Il modello include livelli di ridimensionamento, data augmentation, un backbone MobileNet pre-addestrato, estrazione di patch, pooling globale, e livelli di classificazione.

    - Definizione del modello di fine-tuning: Viene definita una classe PattLiteFineTuneHyperModel per il fine-tuning del modello. Il backbone MobileNet viene parzialmente sbloccato per consentire l'addestramento di alcuni livelli.

    - Ricerca degli iperparametri: Viene utilizzato Keras Tuner per cercare i migliori iperparametri per entrambi i modelli (addestramento e fine-tuning). La ricerca viene eseguita utilizzando la classe RandomSearch e comprende:

        - Modello di Addestramento (PattLiteHyperModel):

            *  units: Numero di unità nel livello Dense (hp.Int('units', min_value=32, max_value=128, step=32))
            * l2_reg: Valore della regularizzazione L2 nel livello Conv2D. (hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='LOG'))
            * l2_reg_2: Valore della regularizzazione L2 nel livello Dense. (hp.Float('l2_reg_2', min_value=1e-5, max_value=1e-2, sampling='LOG'))
            * dropout_rate: Tasso di Dropout nel livello Dense. (hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1) )
            * TRAIN_DROPOUT: Tasso di Dropout dopo il livello GlobalAveragePooling2D. (hp.Float('TRAIN_DROPOUT', min_value=0.2, max_value=0.5, step=0.1))
            * TRAIN_LR: Learning rate dell'ottimizzatore Adam. (hp.Float('TRAIN_LR', min_value=1e-4, max_value=1e-2, sampling='LOG')

        - Modello di Fine-Tuning (PattLiteFineTuneHyperModel))

            * units_ft: Numero di unità nel livello Dense. (hp.Int('units_ft', min_value=32, max_value=128, step=32))
            * l2_reg_fine_tuning: Valore della regularizzazione L2 nel livello Conv2D. (hp.Float('l2_reg_fine_tuning', min_value=1e-5, max_value=1e-2, sampling='LOG')) 
            * l2_reg_FT: Valore della regularizzazione L2 nel livello Dense. (hp.Float('l2_reg_FT', min_value=1e-5, max_value=1e-2, sampling='LOG'))
            * dropout_rate_FT: Tasso di Dropout nel livello Dense. (hp.Float('dropout_rate_FT', min_value=0.3, max_value=0.7, step=0.1))
            * FT_DROPOUT: Tasso di Dropout dopo il livello GlobalAveragePooling2D e SpatialDropout2D. (hp.Float('FT_DROPOUT', min_value=0.2, max_value=0.5, step=0.1)) 
            * FT_LR: Learning rate dell'ottimizzatore Adam. (hp.Float('FT_LR', min_value=1e-6, max_value=1e-2, sampling='LOG'))

    Questi parametri vengono ottimizzati utilizzando la classe RandomSearch di Keras Tuner per trovare i valori che migliorano le prestazioni del modello sui dati di validazione.
    
    - Salvataggio dei migliori iperparametri: I migliori iperparametri trovati durante la ricerca vengono salvati in un file di testo.

2. Questi hyperparameters trovati sono stati applicati nel modello di [patt-lite](../patt-lite/prova.py) modificato, al quale sono stati aggiunte alcune ulteriori modifiche, come: 

    - Regularizzazione L2: Viene applicata la regularizzazione L2 con un valore determinato da best_hps['l2_reg_2']. La regularizzazione L2 aiuta a prevenire l'overfitting penalizzando i pesi grandi, il che può migliorare la generalizzazione del modello. Non avendo regularizzazione, il modello originale potrebbe essere più incline all'overfitting.
    - Dropout: Viene applicato un livello di Dropout con un tasso determinato da best_hps['dropout_rate']. Il Dropout aiuta a prevenire l'overfitting spegnendo casualmente una frazione delle unità durante l'addestramento, il che può migliorare la robustezza del modello.Non avendo Dropout, il modello originale potrebbe essere più incline all'overfitting.


Il codice nuovo è più complesso e include tecniche di regolarizzazione (L2 e Dropout) che possono aiutare a migliorare la generalizzazione del modello e prevenire l'overfitting. Inoltre, il numero di unità nel livello Dense, così come altri parametri, è ottimizzato tramite una ricerca di iperparametri, il che può portare a una migliore performance complessiva del modello.

3. Per cercare di migliorare l'accuratezza del modello, si è inoltre cercato di lavorare sui dataset andando a valutare se le emozioni  fossero sbilanciate, i risultati di questa analisi hanno evidenziato come le varie classi sono sbilanciate tra loro per i dataset considerati, per cui si è preso in considerazione anche l'idea di bilanciare il numero di dati a disposizione utilizzando tecniche di augmentation, in particolare:

    * iaa.Fliplr(0.5): Applica un flip orizzontale alle immagini con una probabilità del 50%. Questo significa che metà delle immagini saranno capovolte orizzontalmente.
    * iaa.Affine(rotate=(-20, 20)): Applica una rotazione affine alle immagini. Le immagini possono essere ruotate di un angolo casuale compreso tra -20 e 20 gradi.
    * iaa.Multiply((0.8, 1.2)): Modifica la luminosità delle immagini moltiplicando i valori dei pixel per un fattore casuale compreso tra 0.8 e 1.2. Questo rende le immagini più scure o più chiare.
    * iaa.GaussianBlur(sigma=(0, 1.0)): Applica una sfocatura gaussiana alle immagini con un valore di sigma casuale compreso tra 0 e 1.0. La sfocatura gaussiana rende le immagini più sfocate.

Tali risultati sono stati salvati sotto il nome di processed_datasetname.h5.
Inoltre, si è valutata anche la possibilità di poter ampliare ancora il dataset con altre immagini (con ulteriori 100 immagini), per cui una seconda prova è stata effettuata aggiungendo ulteriormente immagini, seguendo il processo di data augmentation descritto precedentemente. Questi risultati sono stati salvati sotto il nome di processed_datasetname_5.h5. 
Per questi ulteriori due modelli si è proceduto alla ricerca dei migliori iperparametri come descritto precedentemente. 

4. Una volta cercati questi parametri, si sono effettuate nuovamente delle prove sul nuovo modello con i nuovi parametri.
I risultati di tale analisi sono presenti nella cartella sotto il nome di [Processed_no_augmentation_BestParameters/Bosphorus](./Processed_no_augmentation_BestParameters/Bosphorus) e [Processed_no_augmentation_BestParameters/CK+](./Processed_no_augmentation_BestParameters/CK+).

Da quest'analisi, che ha visto effettuare diverse prove con diversi BATCH_SIZE sia per la fase di training iniziale che per la fase di finetuning, si sono ottenute le seguenti considerazioni:

- Bosphorus: i risultati più promettenti appartengono alle prove con BATCH_SIZE 16 E FT_BATCH_SIZE 32, che si distingue per un buon equilibrio tra accuratezza e perdita, sia sui dati di training che di validation, suggerendo una buona capacità di generalizzazione e un rischio di overfitting contenuto, e con BATCH_SIZE 8 E FT_BATCH_SIZE 64, il cui modello presenta la perdita più bassa sul test set, indicando una potenziale maggiore precisione nelle previsioni. Sulla base dei risultati presentati, **il modello BATCH_SIZE 16 E FT_BATCH_SIZE 32 sembra essere un ottimo candidato** per un ulteriore approfondimento.

- CK+: i risultati più promettenti appartengono alle prove con CK+ TR BATCH 64 E FT BATCH 16 (94.62%), che si distingue per un buon equilibrio tra accuratezza e perdita, sia sui dati di training che di validation. Ciò suggerisce una buona capacità di generalizzazione e un rischio di overfitting contenuto. Anche CK+ TR BATCH 32 E FT BATCH 32 (94.62%) presenta la perdita più bassa sul test set, indicando una potenziale maggiore precisione nelle previsioni.  Sulla base dei risultati presentati, **il modello CK+ TR BATCH 64 E FT BATCH 16 sembra essere un ottimo candidato** per un ulteriore approfondimento.


5. In seguito, si è tentato ancora di migliorare i risultati, lavorando sulle immagini in maniera tale da rendere più facile al modello riconoscere i pattern dei volti e facilizzarne l'apprendimento, riducendo quanto più possibile le "immagini difficili" che il modello non riesce a predire. A tal proposito, è stato usato il codice presente in [classify](../patt-lite/classify.py) che lavora come segue:

- preprocess_image: Preprocessa un'immagine applicando vari passaggi: conversione in scala di grigi, equalizzazione dell'istogramma, sfocatura gaussiana, normalizzazione e conversione in formato BGR.
- identify_difficult_images: Identifica le immagini difficili da classificare per il modello, ovvero quelle per cui la previsione del modello non corrisponde all'etichetta reale.
- find_best_params: Cerca i migliori iperparametri per il modello utilizzando una griglia di parametri (param_grid).
Preprocessa le immagini di addestramento, validazione e test con vari parametri e valuta le prestazioni del modello.
Restituisce i migliori parametri trovati, la migliore accuratezza e il numero minimo di immagini difficili.
- save_data: salva i dati preprocessati in un file HDF5.


6. E' stato infine valutato il modello nuovo con questi nuovi risultati, effettuando diverse prove per diversi valori di TR BATCH SIZE e FT BATCH SIZE, ottenendo:

* Bosphorus:
    - processed_bosphorus_5.h5 con best parameters: ha ottenuto un accuracy di 99.7% con **BATCH_SIZE 32 E FT_BATCH_SIZE 32:**, è il modello che si distingue per le migliori prestazioni complessive, con un'elevata accuratezza e un basso valore di perdita sia sul training set che sul validation set. Ciò indica una buona capacità di generalizzazione e un basso rischio di overfitting. Gli altri modelli presentano prestazioni leggermente inferiori, ma offrono comunque risultati soddisfacenti. Il rischio di overfitting è generalmente basso in tutti i modelli, grazie all'utilizzo di tecniche di regolarizzazione implicite negli algoritmi di ottimizzazione.
    - processed_bosphorus.h5 con best parameters:

* CK+:
    - processed_ckplus_5.h5 con best parameters: ha ottenuto un accuracY del 100% con **BATCH_SIZE 16 E FT_BATCH_SIZE 16**. Questo modello presenta un ottimo equilibrio tra accuratezza, perdita e overfitting. Ha un'accuratezza molto elevata (1.0) e una perdita bassa, senza segni evidenti di overfitting. **BATCH_SIZE 8 E FT_BATCH_SIZE 16** (99.65%) ha un'accuratezza leggermente inferiore rispetto al precedente, ma presenta comunque ottime prestazioni. La perdita è bassa e il rischio di overfitting sembra contenuto.
    - processed_ckplus.h5 con best parameters: 
