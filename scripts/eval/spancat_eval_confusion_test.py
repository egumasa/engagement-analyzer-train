from typing import Optional, List, Dict, Any, Union
from wasabi import Printer

import os
from thinc.api import fix_random_seed
from pathlib import Path
import spacy
import json
import pprint as pp

from spacy.training import Corpus, Example
from spacy.tokens import Doc
from spacy.cli._util import app, Arg, Opt, setup_gpu, import_code
from spacy.scorer import Scorer
from spacy import util
from spacy import displacy

from aligner import needle, water

from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import typer

categories = [
    'ENTERTAIN', "ATTRIBUTE", "ENDORSE", "PRONOUNCE", "CONCUR", "COUNTER",
    "DENY", "MONOGLOSS", "CITATION", "SOURCES", "ENDOPHORIC", "EXEMPLIFYING",
    "EXPOSITORY", "JUSTIFYING", "SUMMATIVE", "COMPARATIVE", "TEXT_SEQUENCING",
    "GOAL_ANNOUNCING", "_"
]

main_categories = [
    'ENTERTAIN', "ATTRIBUTE", "ENDORSE", "PRONOUNCE", "CONCUR", "COUNTER",
    "DENY", "MONOGLOSS", "CITATION", "SOURCES", "ENDOPHORIC", "JUSTIFYING"
]
main_categories = [
    'ENTERTAIN', "ATTRIBUTE", "ENDORSE", "PRONOUNCE", "CONCUR", "COUNTER",
    "DENY", "MONOGLOSS", "JUSTIFYING"
]


def dataset2tags(dev_dataset1,
                 dev_dataset2,
                 nlp,
                 spans_key: str = "sc",
                 pr=False):
    gold_doc = dev_dataset1.reference
    pred_doc = dev_dataset2.reference

    #start a set
    category = set()
    gold_spans = set()
    pred_spans = set()

    for span in gold_doc.spans[spans_key]:
        gold_span = (span.label_, span.start, span.end - 1)
        gold_spans.add(gold_span)
        if span.label_ not in category: category.add(span.label_)

    for span in pred_doc.spans[spans_key]:
        pred_span = (span.label_, span.start, span.end - 1)
        pred_spans.add(pred_span)
        if span.label_ not in category: category.add(span.label_)

    gold_sorted = sorted(gold_spans, key=lambda x: x[1])
    pred_sorted = sorted(pred_spans, key=lambda x: x[1])

    if pr:
        print(gold_sorted)
        print(pred_sorted)

        print()
    if len(gold_sorted) > 0:
        alined = needle(gold_sorted, pred_sorted)
    else:
        alined = ([], [])

    y_gold = [y[0] for y in alined[-2]]
    y_pred = [y[0] for y in alined[-1]]

    if pr:
        for gold, pred in zip(y_gold, y_pred):
            print(gold, pred, sep='\t')
    return (y_gold, y_pred, category)


def evaluate_spancat(model: str,
                     data_path1: Path,
                     data_path2: Path,
                     output: Optional[Path] = None,
                     use_gpu: int = -1,
                     gold_preproc: bool = False,
                     displacy_path: Optional[Path] = None,
                     displacy_limit: int = 25,
                     silent: bool = True,
                     spans_key: str = "sc"):
    getter = getattr
    msg = Printer(no_print=silent, pretty=not silent)
    fix_random_seed()
    setup_gpu(use_gpu, silent=silent)
    data_path1 = util.ensure_path(data_path1)
    data_path2 = util.ensure_path(data_path2)
    output_path = util.ensure_path(output)
    displacy_path = util.ensure_path(displacy_path)
    if not data_path1.exists():
        msg.fail("Evaluation data not found", data_path1, exits=1)
    if not data_path2.exists():
        msg.fail("Evaluation data not found", data_path2, exits=1)
    if displacy_path and not displacy_path.exists():
        msg.fail("Visualization output directory not found",
                 displacy_path,
                 exits=1)

    corpus1 = Corpus(data_path1, gold_preproc=False)
    corpus2 = Corpus(data_path2, gold_preproc=False)

    nlp = util.load_model(model)

    dev_dataset1 = list(corpus1(nlp))
    dev_dataset2 = list(corpus2(nlp))

    category = set()
    tags1 = []
    tags2 = []

    for data1, data2 in zip(dev_dataset1, dev_dataset2):
        anno1, anno2, categ = dataset2tags(data1,
                                           data2,
                                           nlp,
                                           spans_key=spans_key)
        assert len(anno1) == len(anno2)
        tags1.extend(anno1)
        tags2.extend(anno2)
        category.update(categ)

    print("Length of tags: " + str(len(tags1)))

    test_info = {"sentence": 0, "token": 0}

    for data1 in dev_dataset1:
        sentences = data1.split_sents()
        test_info['sentence'] += len(sentences)
        for s in sentences:
            sent_doc = nlp(s.text)
            test_info['token'] += len(sent_doc)

    print(test_info)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=tags1,
        y_pred=tags2,
        labels=list(list(category) + ["_"]).sort(),
        xticks_rotation='vertical',
        cmap=plt.cm.Blues)

    confusion_path = str(output).replace('.json', '.png')
    plt.title("Confusion matrix between two coders")
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')

    print("=============== Classification Results ===============")
    a = classification_report(tags1,
                              tags2,
                              labels=list(category).sort(),
                              digits=4)
    print(a)

    with open(str(output), 'w') as output_f:
        json.dump(a, output_f)

    # kappa
    print("=============== Classification Results ===============")
    kappa = cohen_kappa_score(tags1, tags2, labels=list(category).sort())
    print('Overall cohens kappa: %f' % kappa)
    print()
    # Matthew's correlation coefficient
    mcc = matthews_corrcoef(tags1, tags2)
    print("Matthew's correlation coefficient: %f" % mcc)

    # for cat in list(category):
    #     kappa = cohen_kappa_score(tags1, tags2, labels=list(cat))
    #     print(f'{cat}: %f' % kappa)


def main(
    model: str = Arg(..., help="Model name or path"),
    data_path1: Path = Arg(
        ...,
        help="Location of binary evaluation data in .spacy format",
        exists=True),
    data_path2: Path = Arg(
        ...,
        help="Location of binary evaluation data in .spacy format",
        exists=True),
    output: Optional[Path] = Opt(None,
                                 "--output",
                                 "-o",
                                 help="Output JSON file for metrics",
                                 dir_okay=False),
    code_path: Optional[Path] = Opt(
        None,
        "--code",
        "-c",
        help=
        "Path to Python file with additional code (registered functions) to be imported"
    ),
    use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU"),
    gold_preproc: bool = Opt(False,
                             "--gold-preproc",
                             "-G",
                             help="Use gold preprocessing"),
    displacy_path: Optional[Path] = Opt(
        None,
        "--displacy-path",
        "-dp",
        help="Directory to output rendered parses as HTML",
        exists=True,
        file_okay=False),
    displacy_limit: int = Opt(25,
                              "--displacy-limit",
                              "-dl",
                              help="Limit of parses to render as HTML"),
):
    os.getcwd()
    os.chdir(
        '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Engagement_span_finder'
    )
    # code_path = './scripts/custom_functions.py'
    import_code(code_path)
    # data_path = "data/engagement_spl_dev.spacy"

    evaluate_spancat(
        model,
        data_path1,
        data_path2,
        output=output,
        use_gpu=-1,
        gold_preproc=False,
        displacy_path=displacy_path,
        displacy_limit=25,
        silent=False,
    )


if __name__ == '__main__':
    typer.run(main)
