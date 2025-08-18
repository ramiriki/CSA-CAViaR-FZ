# CSA-CAViaR-FZ
Questa repository contiene i file Python utilizzati per creare i grafici e ottenere i risultati numerici presenti nella tesi di Riccardo Ramina intitolata "Previsione dell’Expected Shortfall e del Value-at-Risk con la Cross-Sectional Aggregation", basata sul lavoro pubblicato da Jie Wang e Yongqiao Wang chiamato "Forecasting Expected Shortfall and Value-at-Risk With Cross-Sectional Aggregation".
Il codice viene messo a disposizione per garantire la trasparenza e la riproducibilità degli esperimenti per coloro i quali volessero analizzarlo e/o migliorarlo.
Nel seguito sono commentati brevemente gli script, spiegando cosa potete trovare al loro interno, dato che commenti più esautivi si trovano al loro interno e la loro produzione segue fedelmente i due lavori sopra citati.

## ak.py
Mostra come variano le densità della distribuzione Beta a seconda dei suoi parametri e produce i grafici del decadimento del coefficiente a_k nei modelli CAViaR a memoria lunga, a seconda dei valori assunti dai parametri della distribuzione Beta.

## ShortLongDecay.py
Decadimenti (iperbolico ed esponenziale) teorici per li modelli CAViaR a memoria lunga e corta.

## CAViaR_DGP.py
Produce grafici per tutti i processi CAViaR a memoria lunga e corta attraverso studi di simulazione.

## acorr_CAViaR.py
Produce i grafici relativi all'autocorrelazione dei processi ottenuti dagli studi di simulazione.

## ParamEst_CAViaR.py
Stima i parametri relativi ai processi prodotti per gli studi di simulazione.



## ExchangeRates.csv
File csv contenente i dati relativi alle 4 serie dei tassi di cambio.

## ERStatistics.py
Contiene il calcolo delle statistiche relative alle serie storiche dei tassi di cambio.

## ERReturn.py
Produce i grafici per le serie storiche dei 4 tassi di cambio e di dei raltivi rendimenti.

## ERHist.py
Produce gli istogrammi delle serie storiche relative ai rendimenti dei 4 tassi di cambio.

## ERAcorr.py
Produce i grafici relativi alle funzioni di autocorrelazione relative ai rendimenti dei 4 tassi di cambio.

## ParamEsr_ER.py
Stima i parametri relativi alle serie storiche relative ai rendimenti dei 4 tassi di cambio.



## StockIndices.csv
File csv contenente i dati relativi alle 4 serie degli indici azionari.

## SIStatistics.py
Contiene il calcolo delle statistiche relative alle serie storiche degli indici azionari.

## SIReturn.py
Produce i grafici per le serie storiche dei 4 indici azionari.

## SIHist.py
Produce gli istogrammi delle serie storiche relative ai 4 indici azionari.

## SIAcorr.py
Produce i grafici relativi alle funzioni di autocorrelazione relative ai 4 indici azionari.
