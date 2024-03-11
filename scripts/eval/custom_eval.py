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

data_path = "data/engagement_spl_dev.spacy"

model = 'packages/en_engagement_spl_RoBERTa_acad_max1_do02_sq_lw-0.2.6.1130/en_engagement_spl_RoBERTa_acad_max1_do02_sq_lw/en_engagement_spl_RoBERTa_acad_max1_do02_sq_lw-0.2.6.1130'

corpus = Corpus(data_path, gold_preproc=False)
nlp = util.load_model(model)
nlp = spacy.load(model)

dev_dataset = list(corpus(nlp))

gold_doc = dev_dataset[0].reference
pred_doc = nlp(dev_dataset[0].text)

#start a set
gold_spans = set()
pred_spans = set()

for span in gold_doc.spans['sc']:
    gold_span = (span.label_, span.start, span.end - 1)
    gold_spans.add(gold_span)

for span in pred_doc.spans['sc']:
    pred_span = (span.label_, span.start, span.end - 1)
    pred_spans.add(pred_span)

alined = needle(sorted(gold_spans), sorted(pred_spans))

y_gold = [y[0] for y in alined[-2]]
y_pred = [y[0] for y in alined[-1]]

from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

cm = confusion_matrix(y_gold, y_pred, labels=categories)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot()
plt.show()

##other scripts
for x, y in zip(gold_doc.spans, pred_doc.spans):
    print(x)
    print

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


score_set(pred_spans, gold_spans)

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