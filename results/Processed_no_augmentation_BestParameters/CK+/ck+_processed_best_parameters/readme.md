
### CK+ without preprocessing but with best hyperparameters

## CK+ TR BATCH 8 E FT BATCH 8

accuracy test set: 0.9032257795333862
accuracy train set: 0.9845201373100281
accuracy validation set: 0.9135135412216187
loss test set: 0.3255533277988434
loss train set: 0.07536045461893082
loss validation set: 0.5857250094413757

## CK+ TR BATCH 8 E FT BATCH 16

accuracy test set: 0.9032257795333862
accuracy train set: 0.9876161217689514
accuracy validation set: 0.8972973227500916
loss test set: 0.6070088744163513
loss train set: 0.058979254215955734
loss validation set: 0.6066474914550781

## CK+ TR BATCH 8 E FT BATCH 32

accuracy test set: 0.9032257795333862
accuracy train set: 0.9969040155410767
accuracy validation set: 0.8918918967247009
loss test set: 0.5433802604675293
loss train set: 0.03315158188343048
loss validation set: 0.5895231366157532

## CK+ TR BATCH 8 E FT BATCH 64

accuracy test set: 0.8924731016159058
accuracy train set: 1.0
accuracy validation set: 0.9189189076423645
loss test set: 0.3564242422580719
loss train set: 0.023734237998723984
loss validation set: 0.3929789662361145

## CK+ TR BATCH 16 E FT BATCH 8

accuracy test set: 0.8709677457809448
accuracy train set: 0.9783281683921814
accuracy validation set: 0.9135135412216187
loss test set: 0.42690497636795044
loss train set: 0.1111581027507782
loss validation set: 0.5352662801742554

## CK+ TR BATCH 16 FT BATCH 16

accuracy test set: 0.9032257795333862
accuracy train set: 0.9891641139984131
accuracy validation set: 0.8972973227500916
loss test set: 0.4022787809371948
loss train set: 0.059868961572647095
loss validation set: 0.6308920979499817

## CK+ TR BATCH 16 E FT BATCH 32

accuracy test set: 0.8924731016159058
accuracy train set: 0.9984520077705383
accuracy validation set: 0.8972973227500916
loss test set: 0.6472594738006592
loss train set: 0.02424893155694008
loss validation set: 0.5827862620353699

## CK+ TR BATCH 16 E FT BATCH 64

accuracy test set: 0.9569892287254333
accuracy train set: 0.9938080310821533
accuracy validation set: 0.9135135412216187
loss test set: 0.2256852239370346
loss train set: 0.03408915176987648
loss validation set: 0.443317174911499

## CK+ TR BATCH 32 E FT BATCH 8

accuracy test set: 0.8387096524238586
accuracy train set: 0.9473684430122375
accuracy validation set: 0.8540540337562561
loss test set: 0.5994337201118469
loss train set: 0.17955949902534485
loss validation set: 0.6465054154396057

## CK+ TR BATCH 32 E FT BATCH 16

accuracy test set: 0.8817204236984253
accuracy train set: 0.9922600388526917
accuracy validation set: 0.8972973227500916
loss test set: 0.5273765325546265
loss train set: 0.06757325679063797
loss validation set: 0.46745216846466064

## CK+ TR BATCH 32 E FT BATCH 32

accuracy test set: 0.9462365508079529
accuracy train set: 0.9969040155410767
accuracy validation set: 0.9189189076423645
loss test set: 0.25277334451675415
loss train set: 0.024008333683013916
loss validation set: 0.4835811257362366

## CK+ TR BATCH 32 E FT BATCH 64

accuracy test set: 0.9247311949729919
accuracy train set: 0.995356023311615
accuracy validation set: 0.9459459185600281
loss test set: 0.34996503591537476
loss train set: 0.027620354667305946
loss validation set: 0.353512704372406

## CK+ TR BATCH 64 E FT BATCH 8

accuracy test set: 0.8924731016159058
accuracy train set: 0.970588207244873
accuracy validation set: 0.8756756782531738
loss test set: 0.6130964756011963
loss train set: 0.12219975143671036
loss validation set: 0.7582345604896545

## CK+ TR BATCH 64 E FT BATCH 16

accuracy test set: 0.9462365508079529
accuracy train set: 0.9876161217689514
accuracy validation set: 0.908108115196228
loss test set: 0.28723588585853577
loss train set: 0.05551852658390999
loss validation set: 0.5129048824310303

## CK+ TR BATCH 64 E FT BATCH 32

accuracy test set: 0.8817204236984253
accuracy train set: 0.9938080310821533
accuracy validation set: 0.908108115196228
loss test set: 0.4491482079029083
loss train set: 0.031023435294628143
loss validation set: 0.48575422167778015

## CK+ TR BATCH 64 E FT BATCH 64

accuracy test set: 0.9139785170555115
accuracy train set: 0.995356023311615
accuracy validation set: 0.9189189076423645
loss test set: 0.3652363419532776
loss train set: 0.02969633974134922
loss validation set: 0.48738980293273926

### Conclusioni

Dall'analisi dei risultati, emergono alcuni modelli promettenti:

* **CK+ TR BATCH 64 E FT BATCH 16:** Questo modello si distingue per un buon equilibrio tra accuratezza e perdita, sia sui dati di training che di validation. Ciò suggerisce una buona capacità di generalizzazione e un rischio di overfitting contenuto.
* **CK+ TR BATCH 32 E FT BATCH 32:** Questo modello presenta la perdita più bassa sul test set, indicando una potenziale maggiore precisione nelle previsioni. Tuttavia, è importante verificare se questa performance è dovuta a un caso fortuito o a una reale capacità del modello.

Sulla base dei risultati presentati, **il modello CK+ TR BATCH 64 E FT BATCH 16 sembra essere un ottimo candidato** per un ulteriore approfondimento.
