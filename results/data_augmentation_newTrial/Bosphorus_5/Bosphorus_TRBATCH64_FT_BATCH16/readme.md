### [Parameters](./parameters.txt)

accuracy test set: 0.983818769454956
accuracy train set: 0.9939675331115723
accuracy validation set: 0.9772727489471436
loss test set: 0.0779852420091629
loss train set: 0.02167990617454052
loss validation set: 0.09493080526590347

F1 Score: 0.9838175163536372
Precision: 0.98395925842302
Recall: 0.9838187702265372
AUC-ROC: 0.999754962019113


1. **Accuracy**:
   - **Test Set**: 0.9838
   - **Train Set**: 0.9940
   - **Validation Set**: 0.9773

   L'accuratezza è molto alta in tutti i set, indicando che il modello è molto preciso nel classificare correttamente i dati. La leggera differenza tra i set di addestramento e di validazione/test suggerisce che il modello generalizza bene.

2. **Loss**:
   - **Test Set**: 0.0780
   - **Train Set**: 0.0217
   - **Validation Set**: 0.0949

   La perdita è bassa, specialmente nel set di addestramento, suggerendo che il modello si adatta bene ai dati. La perdita nel set di validazione è leggermente più alta, ma comunque accettabile, indicando un buon equilibrio tra bias e varianza. La perdita nel set di test è anche abbastanza bassa, confermando la buona performance del modello su dati non visti.

3. **F1 Score**: 0.9838
   - Questo punteggio bilancia precisione e recall, confermando che il modello ha una buona performance complessiva.

4. **Precision**: 0.9840
   - La precisione alta indica che il modello ha pochi falsi positivi, il che è positivo per applicazioni dove i falsi allarmi devono essere minimizzati.

5. **Recall**: 0.9838
   - Un recall alto significa che il modello ha pochi falsi negativi, il che è importante per applicazioni dove è cruciale catturare tutti i casi positivi.

6. **AUC-ROC**: 0.9998
   - Un valore molto vicino a 1, suggerendo che il modello è eccellente nel distinguere tra le classi. Questo indica che il modello ha una capacità molto alta di separare correttamente le classi positive e negative.

In sintesi, i risultati mostrano che il modello è altamente performante, con un'ottima capacità di generalizzazione e una bassa perdita. Anche se c'è un leggero aumento della perdita nel set di validazione rispetto al set di addestramento, i valori sono comunque molto buoni e indicano un modello robusto.

