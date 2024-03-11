import spacy
import json
import pprint as pp

from spacy.training import Corpus, Example
from spacy.tokens import Doc
from spacy.cli._util import app, Arg, Opt, setup_gpu, import_code
from spacy.scorer import Scorer
from spacy import util
from spacy import displacy

from scripts.aligner import needle

categories = [
    'ENTERTAIN', "ATTRIBUTE", "ENDORSE", "PRONOUNCE", "CONCUR", "COUNTER",
    "DENY", "MONOGLOSS", "CITATION", "SOURCES", "ENDOPHORIC", "EXEMPLIFYING",
    "EXPOSITORY", "JUSTIFYING", "SUMMATIVE", "COMPARATIVE", "TEXT_SEQUENCING",
    "GOAL_ANNOUNCING"
]

code_path = './scripts/custom_functions.py'
import_code(code_path)
getter = getattr

data_path1 = "Test_set/20230110_test/ALM.spacy"
data_path2 = "Test_set/20230110_test/RW.spacy"

model = 'packages/en_engagement_spl_RoBERTa_acad_max1_do02_sq_lw-0.2.6.1130/en_engagement_spl_RoBERTa_acad_max1_do02_sq_lw/en_engagement_spl_RoBERTa_acad_max1_do02_sq_lw-0.2.6.1130'

corpus1 = Corpus(data_path1, gold_preproc=False)
corpus2 = Corpus(data_path2, gold_preproc=False)

nlp = util.load_model(model)
nlp = spacy.load(model)

dev_dataset1 = list(corpus1(nlp))
dev_dataset2 = list(corpus2(nlp))


doc1 = dev_dataset1[0].reference
doc2 = dev_dataset2[0].reference
# pred_doc = nlp(dev_dataset[0].text)

#start a set
spans1 = set()
spans2 = set()

for span in doc1.spans['sc']:
    gold_span = (span.label_, span.start, span.end - 1)
    spans1.add(gold_span)

for span in doc2.spans['sc']:
    pred_span = (span.label_, span.start, span.end - 1)
    spans2.add(pred_span)

alined = needle(sorted(spans1), sorted(spans2))

y1 = [y[0] for y in alined[-2]]
y2 = [y[0] for y in alined[-1]]

from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

cm = confusion_matrix(y1, y2, labels=categories)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot()
plt.show()

##other scripts
for x, y in zip(gold_doc.spans, pred_doc.spans):
    print(x)

ents_x2y = dev_dataset[0].get_aligned_spans_x2y(pred_doc.spans['sc'])

for x in ents_x2y:
    print(x.text, x.label_, y.text)

gold_doc.spans['sc']
pred_doc.spans['sc']

ents_x2y = example.get_aligned_spans_x2y(ents_pred)


def score_set(cand: set, gold: set):
    tp = len(cand.intersection(gold))
    fp = len(cand - gold)
    fn = len(gold - cand)
    return tp, fp, fn


score_set(spans2, spans1)

from matplotlib import pyplot
import numpy


def plot_confusion_matrix(docs,
                          classes,
                          normalize=False,
                          cmap=pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    title = 'Confusion Matrix, for SpaCy NER'

    # Compute confusion matrix
    cm = generate_confusion_matrix(docs)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    fig, ax = pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=numpy.arange(cm.shape[1]),
        yticks=numpy.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    pyplot.setp(ax.get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return cm, ax, pyplot


# pred_per_type: Dict[str, Set] = {label: set() for label in labels}

# labels = set([k.label_ for k in getter(gold_doc, '')])

# ents_y2x = dev_dataset[0].get_aligned_spans_x2y(pred_doc, allow_overlap=True)

# for x in ents_y2x:
#     print(x)

# scores = nlp.evaluate(dev_dataset)

# dev_dataset[0]
# gold_doc = dev_dataset[5].reference
# pred_doc = dev_dataset[5].predicted

# pred_doc.ents