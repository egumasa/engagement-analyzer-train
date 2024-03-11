# Distribution of tags

For every dataset I talk about here:
* ATTRIBUTE and ENDORSE tags were merged as ATTRIBUTION
- Their differences are whether the writer "uses" to make their case (ENDORSE) or just report (ATTRIBUTE). 

* PRONOUNCE and CONCUR tags were merged as PROCLAIM



## Three-sentence data

- Three-sentence data had more even distribution of tags across splits.
- These figures show tag counts BEFORE oversampling of minority cases. 

====== train tag counts =====
all:     1390
MONOGLOSS:       2042
ENTERTAIN:       2177
ATTRIBUTE:       834 *
ENDORSE:         100 *(ATTRIBUTE + ENDORSE merged as ATTRIBUTION)
COUNTER:         792
DENY:    700
CONCUR:          96  **
PRONOUNCE:       221 **(CONCUR + PRONOUNCE merged as PROCLAIM)
CITATION:        413
SOURCES:         646
ENDOPHORIC:      162
JUSTIFYING:      726

====== dev tag counts =====
all:     174
MONOGLOSS:       286
ENTERTAIN:       266
ATTRIBUTE:       111 *
ENDORSE:         19  *
COUNTER:         110
DENY:    85
CONCUR:          12  **
PRONOUNCE:       39  **
CITATION:        68
SOURCES:         83
ENDOPHORIC:      19
JUSTIFYING:      105

====== test =====
all:     174
MONOGLOSS:       261
ENTERTAIN:       227
ATTRIBUTE:       111 *
ENDORSE:         18  *
COUNTER:         93
DENY:    71
CONCUR:          14  **
PRONOUNCE:       35  **
CITATION:        102
SOURCES:         95
ENDOPHORIC:      19
JUSTIFYING:      93

## Three-sentence + paragraph data

====== train =====
all:     1222
MONOGLOSS:       2057
ENTERTAIN:       2153
ATTRIBUTE:       850 *
ENDORSE:         115 *
COUNTER:         799
DENY:    687
CONCUR:          104 **
PRONOUNCE:       241 **
CITATION:        471
SOURCES:         641
ENDOPHORIC:      164
JUSTIFYING:      765

====== dev =====
all:     153
MONOGLOSS:       268
ENTERTAIN:       261
ATTRIBUTE:       106 *
ENDORSE:         9   *
COUNTER:         111
DENY:    73
CONCUR:          14  **
PRONOUNCE:       25  **
CITATION:        54
SOURCES:         94
ENDOPHORIC:      19
JUSTIFYING:      80

====== test =====
all:     153
MONOGLOSS:       264
ENTERTAIN:       256
ATTRIBUTE:       100 *
ENDORSE:         13  *
COUNTER:         85
DENY:    96
CONCUR:          4   **
PRONOUNCE:       29  **
CITATION:        58
SOURCES:         89
ENDOPHORIC:      17
JUSTIFYING:      79

# Three-sentence dataset

## Dual Transformer model

### Development
=============== Dev Results ===============
              precision    recall  f1-score   support

   MONOGLOSS       0.80      0.80      0.80       966
   ENTERTAIN       0.85      0.87      0.86      1102
 ATTRIBUTION       0.78      0.81      0.79       872
     COUNTER       0.93      0.89      0.91       470
        DENY       0.88      0.75      0.81       298
    PROCLAIM       0.87      0.65      0.75       452
    CITATION       0.94      0.89      0.92       409
     SOURCES       0.71      0.71      0.71       532
  ENDOPHORIC       0.75      0.71      0.73       185
  JUSTIFYING       0.72      0.75      0.73       720

   micro avg       0.81      0.80      0.80      6006
   macro avg       0.82      0.78      0.80      6006
weighted avg       0.81      0.80      0.80      6006

Overall cohens kappa: 0.901148

### Test
* f1-change indices changes in f1 score from Dev set

=============== Test Results ===============
              precision    recall  f1-score   support f1-change

   MONOGLOSS       0.70      0.84      0.77       788 (-3 pts)
   ENTERTAIN       0.80      0.81      0.81       834 (-5 pts)
 ATTRIBUTION       0.75      0.69      0.72       864 (-7 pts)
     COUNTER       0.85      0.86      0.85       438 (-6 pts)
        DENY       0.89      0.77      0.83       326 (+2 pts)
    PROCLAIM       0.80      0.59      0.68       441 (-8 pts)
    CITATION       0.92      0.90      0.91       493 (-1 pts)
     SOURCES       0.78      0.63      0.70       621 (-1 pts)
  ENDOPHORIC       0.63      0.67      0.65       150 (-8 pts)
  JUSTIFYING       0.83      0.82      0.82       732 (+9 pts)

   micro avg       0.79      0.77      0.78      5687
   macro avg       0.79      0.76      0.77      5687  (F1: -3 points)
weighted avg       0.80      0.77      0.78      5687

Overall cohens kappa: 0.890304


## RoBERTa + biLSTM

### Development
=============== Dev Results ===============
              precision    recall  f1-score   support

   MONOGLOSS       0.83      0.79      0.81       966
   ENTERTAIN       0.88      0.81      0.84      1102
 ATTRIBUTION       0.78      0.77      0.78       872
     COUNTER       0.84      0.84      0.84       470
        DENY       0.84      0.74      0.79       298
    PROCLAIM       0.96      0.57      0.71       452
    CITATION       0.94      0.91      0.93       409
     SOURCES       0.69      0.65      0.67       532
  ENDOPHORIC       0.84      0.74      0.79       185
  JUSTIFYING       0.80      0.72      0.76       720

   micro avg       0.83      0.76      0.79      6006
   macro avg       0.84      0.75      0.79      6006
weighted avg       0.84      0.76      0.79      6006

Overall cohens kappa: 0.892544

### Test
=============== Test Results ===============
              precision    recall  f1-score   support

   MONOGLOSS       0.77      0.78      0.77       788 (-4 pts)
   ENTERTAIN       0.86      0.79      0.83       834 (-1 pts)
 ATTRIBUTION       0.74      0.64      0.68       864 (-10 pts)
     COUNTER       0.80      0.93      0.86       438 (+2 pts)
        DENY       0.88      0.81      0.84       326 (+5 pts)
    PROCLAIM       0.84      0.55      0.66       441 (-5 pts)
    CITATION       0.91      0.90      0.91       493 (-2 pts)
     SOURCES       0.79      0.68      0.73       621 (+5 pts)
  ENDOPHORIC       0.83      0.69      0.75       150 (-4 pts)
  JUSTIFYING       0.80      0.67      0.73       732 (-3 pts)

   micro avg       0.81      0.74      0.77      5687
   macro avg       0.82      0.74      0.78      5687
weighted avg       0.81      0.74      0.77      5687

Overall cohens kappa: 0.883565

## RoBERTa model (closest to default SpaCy setting)

=============== Dev Results ===============
              precision    recall  f1-score   support

   MONOGLOSS       0.79      0.82      0.81       966
   ENTERTAIN       0.87      0.76      0.81      1102
 ATTRIBUTION       0.83      0.76      0.79       872
     COUNTER       0.86      0.65      0.74       470
        DENY       0.66      0.77      0.71       298
    PROCLAIM       0.85      0.71      0.77       452
    CITATION       0.90      0.92      0.91       409
     SOURCES       0.70      0.67      0.68       532
  ENDOPHORIC       0.45      0.81      0.58       185
  JUSTIFYING       0.78      0.70      0.73       720

   micro avg       0.79      0.75      0.77      6006
   macro avg       0.77      0.75      0.75      6006
weighted avg       0.80      0.75      0.77      6006

Overall cohens kappa: 0.875524

=============== Test Results ===============
              precision    recall  f1-score   support

   MONOGLOSS       0.72      0.85      0.78       788 (-3 pts)
   ENTERTAIN       0.81      0.72      0.76       834 (-5 pts)
 ATTRIBUTION       0.72      0.68      0.70       864 (-9 pts)
     COUNTER       0.87      0.66      0.75       438 (+1 pts)
        DENY       0.81      0.88      0.84       326 (+13 pts)
    PROCLAIM       0.79      0.57      0.66       441 (-11 pts)
    CITATION       0.90      0.87      0.89       493 (-2 pts)
     SOURCES       0.74      0.68      0.71       621 (+3 pts)
  ENDOPHORIC       0.46      0.75      0.57       150 (-1 pts)
  JUSTIFYING       0.85      0.73      0.78       732 (+5 pts)

   micro avg       0.77      0.74      0.75      5687
   macro avg       0.77      0.74      0.75      5687
weighted avg       0.78      0.74      0.75      5687

Overall cohens kappa: 0.858103

# Three-sentence + Paragraph dataset

## Dual Transformer 

=============== Dev Results ===============
              precision    recall  f1-score   support

   MONOGLOSS       0.83      0.78      0.81       832
   ENTERTAIN       0.83      0.86      0.84       890
 ATTRIBUTION       0.68      0.81      0.74       694
     COUNTER       0.88      0.87      0.88       413
        DENY       0.90      0.75      0.82       279
    PROCLAIM       0.81      0.62      0.70       387
    CITATION       0.96      0.97      0.96       312
     SOURCES       0.74      0.71      0.72       517
  ENDOPHORIC       0.74      0.83      0.78       168
  JUSTIFYING       0.83      0.81      0.82       554

   micro avg       0.81      0.80      0.80      5046
   macro avg       0.82      0.80      0.81      5046
weighted avg       0.81      0.80      0.80      5046

Overall cohens kappa: 0.899490

=============== Test Results ===============
              precision    recall  f1-score   support

   MONOGLOSS       0.83      0.72      0.77       839 (-4 pts)
   ENTERTAIN       0.80      0.76      0.78      1057 (-6 pts)
 ATTRIBUTION       0.67      0.66      0.67       742 (-7 pts)
     COUNTER       0.87      0.86      0.87       366 (-1 pts)
        DENY       0.97      0.86      0.91       452 (+9 pts)
    PROCLAIM       0.62      0.44      0.52       284 (-18pts)
    CITATION       0.95      1.00      0.97       228 (+2 pts)
     SOURCES       0.75      0.82      0.79       535 (+7 pts)
  ENDOPHORIC       0.66      0.79      0.72       203 (+6 pts)
  JUSTIFYING       0.83      0.75      0.79       655 (-3 pts)

   micro avg       0.80      0.76      0.77      5361
   macro avg       0.80      0.77      0.78      5361 (F1: - 3 points)
weighted avg       0.80      0.76      0.77      5361

Overall cohens kappa: 0.891672


## RoBERTa + biLSTM

### Dev 
=============== Dev Results ===============
              precision    recall  f1-score   support

   MONOGLOSS       0.79      0.79      0.79       832
   ENTERTAIN       0.84      0.70      0.76       890
 ATTRIBUTION       0.60      0.80      0.68       694
     COUNTER       0.93      0.67      0.78       413
        DENY       0.82      0.86      0.84       279
    PROCLAIM       0.93      0.56      0.70       387
    CITATION       0.98      0.98      0.98       312
     SOURCES       0.75      0.82      0.78       517
  ENDOPHORIC       0.69      0.69      0.69       168
  JUSTIFYING       0.78      0.74      0.76       554

   micro avg       0.78      0.76      0.77      5046
   macro avg       0.81      0.76      0.78      5046
weighted avg       0.80      0.76      0.77      5046

Overall cohens kappa: 0.871921

### Test

=============== Test Results ===============
              precision    recall  f1-score   support

   MONOGLOSS       0.84      0.76      0.80       839 (+1 pt)
   ENTERTAIN       0.88      0.66      0.75      1057 (-1 pt)
 ATTRIBUTION       0.60      0.82      0.69       742 (+1 pt)
     COUNTER       0.82      0.64      0.72       366 (-6 pts)
        DENY       0.93      0.90      0.91       452 (+7 pts)
    PROCLAIM       0.85      0.37      0.51       284 (-19 pts)
    CITATION       0.91      1.00      0.95       228 (-3 pts)
     SOURCES       0.67      0.79      0.72       535 (-6 pts)
  ENDOPHORIC       0.99      0.75      0.85       203 (+16 pts)
  JUSTIFYING       0.88      0.56      0.68       655 (-8 pts)

   micro avg       0.79      0.72      0.75      5361
   macro avg       0.84      0.72      0.76      5361
weighted avg       0.82      0.72      0.75      5361

Overall cohens kappa: 0.860739