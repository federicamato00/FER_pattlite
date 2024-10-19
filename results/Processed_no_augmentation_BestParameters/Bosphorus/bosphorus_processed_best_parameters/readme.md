
#### Bosphorus without processing but with best hyperparameters

## Modello BATCH_SIZE 8 E FT_BATCH_SIZE 8

accuracy test set: 0.7058823704719543
accuracy train set: 0.9915682673454285
accuracy validation set: 0.8117647171020508
loss test set: 1.912962794303894
loss train set: 0.03717033565044403
loss validation set: 1.5750055313110352

## Modello BATCH_SIZE 8 E FT_BATCH_SIZE 16

accuracy test set: 0.7882353067398071
accuracy train set: 0.994940996170044
accuracy validation set: 0.8294117450714111
loss test set: 1.4464211463928223
loss train set: 0.03499799221754074
loss validation set: 1.229645013809204


## Modello BATCH_SIZE 8 E FT_BATCH_SIZE 32

accuracy test set: 0.7764706015586853
accuracy train set: 0.9898819327354431
accuracy validation set: 0.7941176295280457
loss test set: 1.36195707321167
loss train set: 0.04447163641452789
loss validation set: 1.0095815658569336


## Modello BATCH_SIZE 8 E FT_BATCH_SIZE 64

accuracy test set: 0.7882353067398071
accuracy train set: 0.9932546615600586
accuracy validation set: 0.8294117450714111
loss test set: 1.0111397504806519
loss train set: 0.055947817862033844
loss validation set: 0.9332576990127563

## Modello BATCH_SIZE 16 E FT_BATCH_SIZE 8

accuracy test set: 0.8117647171020508
accuracy train set: 0.9915682673454285
accuracy validation set: 0.8058823347091675
loss test set: 1.9212111234664917
loss train set: 0.045585472136735916
loss validation set: 1.6230157613754272

## Modello BATCH_SIZE 16 E FT_BATCH_SIZE 16

accuracy test set: 0.7647058963775635
accuracy train set: 0.9932546615600586
accuracy validation set: 0.8235294222831726
loss test set: 1.6031856536865234
loss train set: 0.04068312793970108
loss validation set: 1.1285983324050903

## Modello BATCH_SIZE 16 E FT_BATCH_SIZE 32

accuracy test set: 0.8352941274642944
accuracy train set: 0.994940996170044
accuracy validation set: 0.8294117450714111
loss test set: 1.2454429864883423
loss train set: 0.04584769904613495
loss validation set: 1.1319024562835693

## Modello BATCH_SIZE 16 E FT_BATCH_SIZE 64

accuracy test set: 0.7647058963775635
accuracy train set: 0.9865092635154724
accuracy validation set: 0.8235294222831726
loss test set: 1.047795057296753
loss train set: 0.07099002599716187
loss validation set: 0.8752664923667908

## Modello BATCH_SIZE 32 E FT_BATCH_SIZE 8

accuracy test set: 0.7647058963775635
accuracy train set: 0.9881955981254578
accuracy validation set: 0.800000011920929
loss test set: 2.045854330062866
loss train set: 0.05504511669278145
loss validation set: 1.8766975402832031

## Modello BATCH_SIZE 32 E FT_BATCH_SIZE 16

accuracy test set: 0.7647058963775635
accuracy train set: 0.9898819327354431
accuracy validation set: 0.8294117450714111
loss test set: 1.504422664642334
loss train set: 0.04437389597296715
loss validation set: 1.1964111328125

## Modello BATCH_SIZE 32 E FT_BATCH_SIZE 32

accuracy test set: 0.7882353067398071
accuracy train set: 0.9932546615600586
accuracy validation set: 0.8058823347091675
loss test set: 1.0969743728637695
loss train set: 0.0411369763314724
loss validation set: 1.119679570198059

## Modello BATCH_SIZE 32 E FT_BATCH_SIZE 64

accuracy test set: 0.7764706015586853
accuracy train set: 0.994940996170044
accuracy validation set: 0.8294117450714111
loss test set: 1.0421141386032104
loss train set: 0.05754910781979561
loss validation set: 0.7733616828918457

## Modello BATCH_SIZE 64 E FT_BATCH_SIZE 8

accuracy test set: 0.7764706015586853
accuracy train set: 0.9898819327354431
accuracy validation set: 0.8294117450714111
loss test set: 1.9386526346206665
loss train set: 0.05019594728946686
loss validation set: 1.6726833581924438

## Modello BATCH_SIZE 64 E FT_BATCH_SIZE 16

accuracy test set: 0.7764706015586853
accuracy train set: 0.994940996170044
accuracy validation set: 0.7764706015586853
loss test set: 1.5446652173995972
loss train set: 0.029404455795884132
loss validation set: 1.3768547773361206

## Modello BATCH_SIZE 64 E FT_BATCH_SIZE 32

accuracy test set: 0.7529411911964417
accuracy train set: 0.9915682673454285
accuracy validation set: 0.8117647171020508
loss test set: 1.3172309398651123
loss train set: 0.049790166318416595
loss validation set: 0.9483112096786499

## Modello BATCH_SIZE 64 E FT_BATCH_SIZE 64

accuracy test set: 0.800000011920929
accuracy train set: 0.9932546615600586
accuracy validation set: 0.7882353067398071
loss test set: 1.1639351844787598
loss train set: 0.05688805133104324
loss validation set: 1.0171709060668945

### CONCLUSIONI 

Dall'analisi dei risultati, emergono alcuni modelli promettenti:

* **BATCH_SIZE 16 E FT_BATCH_SIZE 32:** Questo modello si distingue per un buon equilibrio tra accuratezza e perdita, sia sui dati di training che di validation. Ciò suggerisce una buona capacità di generalizzazione e un rischio di overfitting contenuto.
* **BATCH_SIZE 8 E FT_BATCH_SIZE 64:** Questo modello presenta la perdita più bassa sul test set, indicando una potenziale maggiore precisione nelle previsioni.

Sulla base dei risultati presentati, **il modello BATCH_SIZE 16 E FT_BATCH_SIZE 32 sembra essere un ottimo candidato** per un ulteriore approfondimento.

