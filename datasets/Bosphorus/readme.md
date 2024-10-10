# Bosphorus Data Augmentation Dataset

## Introduzione
Il progetto utilizza tecniche di data augmentation per migliorare la qualità e la quantità dei dati disponibili nel dataset Bosphorus. La data augmentation è una tecnica che permette di generare nuovi dati a partire da quelli esistenti, applicando trasformazioni come rotazioni, traslazioni, scaling, ecc.
Il file data_prova.py viene usato per raccogliere le immagini dal dataset, processarle con tecniche di data augmentation e poi salvarle in un file .h5 pronto per essere usato in classify.h5

## File del Progetto

### bosphorus_data_augmentation.h5
Questo file contiene le informazioni del dataset processato con tecniche di data augmentation, tra cui:

- **iaa.Fliplr(0.5)**: Questo metodo esegue un flip orizzontale (ribaltamento da sinistra a destra) sulle immagini. Il parametro 0.5 indica la probabilità con cui ogni immagine verrà ribaltata. In questo caso, c'è una probabilità del 50% che ogni immagine venga ribaltata orizzontalmente.

- **iaa.Affine(rotate=(-20, 20))**: uesto metodo applica una trasformazione affine alle immagini, in particolare una rotazione.
Il parametro rotate=(-20, 20) specifica che le immagini possono essere ruotate di un angolo casuale compreso tra -20 e 20 gradi.

- **iaa.Multiply((0.8, 1.2))**: Questo metodo modifica la luminosità delle immagini. Il parametro (0.8, 1.2) indica che i valori dei pixel delle immagini possono essere moltiplicati per un fattore casuale compreso tra 0.8 e 1.2. Un valore inferiore a 1 scurisce l'immagine, mentre un valore superiore a 1 la schiarisce.

- **iaa.GaussianBlur(sigma=(0, 1.0))**: Questo metodo applica un blur gaussiano alle immagini. Il parametro sigma=(0, 1.0) specifica che il grado di sfocatura (sigma) può variare tra 0 e 1.0. Un valore di sigma più alto produce un effetto di sfocatura più marcato.


### bosphorus_data_augmentation_2.h5
Anche questo file contiene un dataset processato con tecniche di data augmentation in modo da migliorare non solo la distribuzione del dataset che altrimenti risulterebbe sbilanciato, ma anche per aumentare il numero di immagini utili per l'addestramento successivo.
Le tecniche di data augmentation sono:
- **iaa.Fliplr(0.5)**: Esegue un flip orizzontale (ribaltamento da sinistra a destra) sulle immagini.
Il parametro 0.5 indica la probabilità con cui ogni immagine verrà ribaltata. In questo caso, c'è una probabilità del 50% che ogni immagine venga ribaltata orizzontalmente.

- **iaa.Affine(rotate=(-20, 20))**:  Applica una trasformazione affine alle immagini, in particolare una rotazione.
Il parametro rotate=(-20, 20) specifica che le immagini possono essere ruotate di un angolo casuale compreso tra -20 e 20 gradi.

- **iaa.Multiply((0.8, 1.2))** Modifica la luminosità delle immagini. Il parametro (0.8, 1.2) indica che i valori dei pixel delle immagini possono essere moltiplicati per un fattore casuale compreso tra 0.8 e 1.2. Un valore inferiore a 1 scurisce l'immagine, mentre un valore superiore a 1 la schiarisce.

- **iaa.GaussianBlur(sigma=(0, 1.0))**:  Applica un blur gaussiano alle immagini. Il parametro sigma=(0, 1.0) specifica che il grado di sfocatura (sigma) può variare tra 0 e 1.0. Un valore di sigma più alto produce un effetto di sfocatura più marcato.

- **iaa.TranslateX(percent=(-0.2, 0.2))**: Trasla (sposta) le immagini orizzontalmente. Il parametro percent=(-0.2, 0.2) indica che le immagini possono essere traslate orizzontalmente di una percentuale casuale compresa tra -20% e 20% della larghezza dell'immagine.

- **iaa.TranslateY(percent=(-0.2, 0.2))**: Trasla (sposta) le immagini verticalmente. Il parametro percent=(-0.2, 0.2) indica che le immagini possono essere traslate verticalmente di una percentuale casuale compresa tra -20% e 20% dell'altezza dell'immagine.

- **iaa.ScaleX((0.8, 1.2))**  Scala (zoom) le immagini orizzontalmente.Il parametro (0.8, 1.2) indica che le immagini possono essere scalate orizzontalmente di un fattore casuale compreso tra 0.8 e 1.2.

- **iaa.ScaleY((0.8, 1.2))**: Scala (zoom) le immagini verticalmente. Il parametro (0.8, 1.2) indica che le immagini possono essere scalate verticalmente di un fattore casuale compreso tra 0.8 e 1.2.

- **iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))**: 
Aggiunge rumore gaussiano alle immagini. Il parametro scale=(0, 0.05*255) specifica che il rumore aggiunto può avere una deviazione standard casuale compresa tra 0 e 0.05*255.

- **iaa.Cutout(nb_iterations=1, size=0.2, squared=True)**:  Rimuove (taglia) una porzione dell'immagine. I parametri:
    1. nb_iterations=1: Numero di volte che l'operazione di cutout viene eseguita.
    2. size=0.2: Dimensione della porzione rimossa, espressa come percentuale dell'immagine.
    3. squared=True: Indica che la porzione rimossa è di forma quadrata.

- **iaa.ElasticTransformation(alpha=50, sigma=5)**: Applica una trasformazione elastica alle immagini, deformandole in modo non lineare. I parametri:
    1. alpha=50: Intensità della deformazione.
    2. sigma=5: Larghezza del kernel gaussiano utilizzato per la deformazione.


### bosphorus_data_augmentation_3.h5
In questo file, sono state utilizzate le tecniche precedentemente usate anche per bosphorus_data_augmentation_2.h5 eliminando però la tecnica di cutout e di ElasticTrasformation per evitare deformazioni troppo elevate.

### bosphorus_data_augmentation_4.h5

In questo file sono state utilizzate le tecniche usate in bosphorus_data_augmentation_3.h5, tuttavia oltre che cercare di bilanciare il dataset, sono state aggiunte più immagini in modo da aumentare il numero di immagini prese in esame per l'addestramento del modello. Vengono aggiunte 100 immagini per ogni classe.

### bosphorus_data_augmentation_5.h5

In questo file sono state utilizzate le tecniche usate in bosphorus_data_augmentation.h5, tuttavia oltre che cercare di bilanciare il dataset, sono state aggiunte più immagini in modo da aumentare il numero di immagini prese in esame per l'addestramento del modello. Vengono aggiunte 100 immagini per ogni classe.


Queste trasformazioni vengono applicate in sequenza alle immagini, permettendo di aumentare la varietà del dataset e migliorare la robustezza del modello addestrato.