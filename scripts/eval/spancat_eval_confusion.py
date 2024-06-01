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
from spacy.tokens import SpanGroup

from aligner import needle, water

from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import typer

import pandas as pd
from collections import Counter

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


def del_spans(span_sc, indexes: list):

    indexes.sort(
        reverse=True
    )  # reversing allows the deletion from the last, keeping the original index

    for idx in indexes:
        if idx + 1 < len(span_sc):
            del span_sc[idx + 1]


def delete_overlapping_span(span_sc: dict):
    # print(span_sc)
    start_token_list = [spn.start for spn in span_sc]
    dict_ = Counter(start_token_list)
    overlap = {k: v for k, v in dict_.items() if v > 1}

    id_del = []
    id_comp = {}

    info = {}
    for n, (spn, score) in enumerate(zip(span_sc, span_sc.attrs['scores']),
                                     start=0):
        res = {
            'score': score,
            'spn': spn,
            'label': spn.label_,
            'start': spn.start,
            'end': spn.end,
            'compare': spn.start in overlap,
            "sents": len(list(spn.sents))
        }
        # print(res)
        info[n] = res

        if res['compare']:
            if spn.start not in id_comp:
                id_comp[spn.start] = n
            else:
                same_lbl = res['label'] == info[id_comp[spn.start]]['label']
                update = res['score'] > info[id_comp[spn.start]]['score']
                if update and same_lbl:
                    print(res['label'], info[id_comp[spn.start]]['label'])
                    print(same_lbl)
                    id_del.append(id_comp[spn.start])
                    id_comp[spn.start] = n
                else:
                    id_del.append(n)
                # print(update)

        # delete span beyond sentences
        if len(list(spn.sents)) > 1:
            id_del.append(n)

    # print(id_comp)
    del_spans(span_sc, id_del)
    # for n, idx in enumerate(id_del):
    #     # print(idx)

    #     try:
    #         del span_sc[idx - n]
    #     except IndexError:
    #         continue


def cleanup_justify(doc, span_sc: dict):
    # This function adjusts the JUSTIFYING span

    # First create an index of span with JUSTIFYING tags
    justifies = {}
    for idx, span in enumerate(span_sc):
        # temp_root = span.root
        # while span.start <= temp_root.head.i <= span.end:
        #     temp_root = temp_root.head
        if span.label_ in ['JUSTIFYING']:
            justifies[span.root] = {
                "span": span,
                "head": span.root.head,
                "start": span.start,
                "end": span.end,
                "del": False,
                "dependent": False,
                "span_idx": idx
            }
    # print(justifies)

    # flagging the dependency
    for spanroot, info in justifies.items():
        if spanroot.head in justifies:
            info['dependent'] = True
            info['del'] = True

    # print(justifies)
    new_spans = []
    for spanroot, info in justifies.items():

        if not info['dependent']:
            # print("New Justifying candidate span:")
            # print(doc[spanroot.left_edge.i:spanroot.right_edge.i + 1])

            new_span = doc[spanroot.left_edge.i:spanroot.right_edge.i + 1]
            new_span.label_ = "JUSTIFYING"

            if new_span not in span_sc:
                new_spans.append(new_span)
                info['del'] = True

        else:
            info['del'] = True

    to_delete = [
        info['span_idx'] for spanroot, info in justifies.items() if info['del']
    ]

    to_delete_span = [
        info['span'] for spanroot, info in justifies.items() if info['del']
    ]

    # print(to_delete)
    # print(to_delete_span)

    del_spans(span_sc, to_delete)

    span_grp = SpanGroup(doc, spans=new_spans)
    span_sc.extend(span_grp)

    # print(justifies)


def dataset2tags(dev_dataset_sub, nlp, spans_key: str = "sc", pr=False):
    gold_doc = dev_dataset_sub.reference
    pred_doc = nlp(dev_dataset_sub.text)

    cleanup_justify(pred_doc, pred_doc.spans['sc'])
    # delete_overlapping_span(pred_doc.spans['sc'])

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
                     data_path: Path,
                     output: Optional[Path] = None,
                     use_gpu: int = -1,
                     gold_preproc: bool = False,
                     displacy_path: Optional[Path] = None,
                     displacy_limit: int = 100,
                     silent: bool = True,
                     spans_key: str = "sc",
                     return_res: bool = False,
                    parser = False,
                    ner = False
                     ):
    getter = getattr
    msg = Printer(no_print=silent, pretty=not silent)
    fix_random_seed()
    setup_gpu(use_gpu, silent=silent)
    data_path = util.ensure_path(data_path)
    output_path = util.ensure_path(output)
    displacy_path = util.ensure_path(displacy_path)
    if not data_path.exists():
        msg.fail("Evaluation data not found", data_path, exits=1)
    if displacy_path and not displacy_path.exists():
        msg.fail("Visualization output directory not found",
                 displacy_path,
                 exits=1)

    if "model-best" in model:
        name = model.split("/")[-2]
    else:
        name = model.split("/")[-1]

    corpus = Corpus(data_path, gold_preproc=False)
    # nlp = util.load_model(model)
    nlp = spacy.load(model)
    # nlp = spacy.load("en_engagement_spl_RoBERTa_acad_db")
    dev_dataset = list(corpus(nlp))

    category = set()
    gold_tags = []
    pred_tags = []

    for data in dev_dataset:
        gold, pred, categ = dataset2tags(data, nlp, spans_key=spans_key)
        assert len(gold) == len(pred)
        gold_tags.extend(gold)
        pred_tags.extend(pred)
        category.update(categ)

    disp = ConfusionMatrixDisplay.from_predictions(y_true=gold_tags,
                                                   y_pred=pred_tags,
                                                   labels=list(category) +
                                                   ["_"],
                                                   xticks_rotation='vertical',
                                                   cmap=plt.cm.Blues)

    confusion_path = str(output).replace('.json', '')
    plt.title(confusion_path.split(os.path.sep)[-1])
    plt.savefig(f"{confusion_path}_{name}_test.png", dpi=300)

    print("=============== Classification Results ===============")
    a = classification_report(gold_tags, pred_tags, labels=list(category))
    print(a)

    # kappa
    print("=============== Cohen's kappa Results ================")
    kappa = cohen_kappa_score(gold_tags, pred_tags, labels=list(category))
    print('Overall cohens kappa: %f' % kappa)

    # kappa
    print(
        "=============== The Matthews correlation coefficient Results ================"
    )
    mcc = matthews_corrcoef(gold_tags, pred_tags)
    print('Matthews Correlation Coefficient: %f' % mcc)

    print("=============== Classification Results (more) ===============")
    a = classification_report(gold_tags,
                              pred_tags,
                              labels=list(category).sort(),
                              digits=4)
    print(a)

    res_dict = classification_report(gold_tags,
                                     pred_tags,
                                     labels=list(category).sort(),
                                     output_dict=True)

    # kappa
    kappa = cohen_kappa_score(gold_tags,
                              pred_tags,
                              labels=list(category).sort())
    print('Overall cohens kappa: %f' % kappa)

    res_dict['kappa'] = {'Overall': kappa}
    res_dict["MCC"] = {'Overall': mcc}

    with open(f"{output_path}_{name}_test.json", 'w') as f:
        json.dump(res_dict, f, indent=2)

    df = pd.DataFrame.from_dict(res_dict)
    # df.to_csv("test_df_columns.csv")
    df.transpose().to_csv(f"{output_path}_{name}_index.csv")

    if displacy_path:
        factory_names = [
            nlp.get_pipe_meta(pipe).factory for pipe in nlp.pipe_names
        ]
        docs = list(
            nlp.pipe(ex.reference.text for ex in dev_dataset[:displacy_limit]))
        if parser:
            render_deps = "parser" in factory_names
        if ner:
            render_ents = "ner" in factory_names
        render_spans = "spancat" in factory_names
        if parser and ner:
            render_parses(
                docs,
                displacy_path,
                model_name=model,
                limit=displacy_limit,
                deps=render_deps,
                ents=render_ents,
                spans=render_spans,
            )
        else:
            render_parses(
                docs,
                displacy_path,
                model_name=model,
                limit=displacy_limit,
                # deps=render_deps,
                # ents=render_ents,
                spans=render_spans,
            )
        msg.good(f"Generated {displacy_limit} parses as HTML", displacy_path)

    if output_path is not None:
        srsly.write_json(output_path, data)
        msg.good(f"Saved results to {output_path}")

    if return_res:
        return res_dict


def render_parses(
    docs: List[Doc],
    output_path: Path,
    model_name: str = "",
    limit: int = 250,
    deps: bool = True,
    ents: bool = True,
    spans: bool = True,
):
    docs[0].user_data["title"] = model_name
    if ents:
        html = displacy.render(docs[:limit], style="ent", page=True)
        with (output_path / "entities.html").open("w",
                                                  encoding="utf8") as file_:
            file_.write(html)
    if deps:
        html = displacy.render(docs[:limit],
                               style="dep",
                               page=True,
                               options={"compact": True})
        with (output_path / "parses.html").open("w", encoding="utf8") as file_:
            file_.write(html)
    if spans:
        html = displacy.render(docs[:limit], style="span", page=True)
        with (output_path / "spans.html").open("w", encoding="utf8") as file_:
            file_.write(html)


# len(nlp(dev_dataset[2].text.strip()))

# len(dev_dataset[2].reference)

# for x, y in zip(dev_dataset[2].reference, nlp(dev_dataset[2].text.strip())):
#     print(x, y)

# %% [markdown]
# # Creating both gold-standard and predicted data

# %%

# dataset2tags(dev_dataset[2])

# dataset2tags(dev_dataset[3])
# dataset2tags(dev_dataset[7])

# gold_tags = []
# pred_tags = []

# for data in dev_dataset:
#     gold, pred = dataset2tags(data)
#     assert len(gold) == len(pred)
#     gold_tags.extend(gold)
#     pred_tags.extend(pred)

# %%
# cm = confusion_matrix(gold_tags, pred_tags, labels=categories)

# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=categories,
# )

# disp.plot(xticks_rotation='horizontal')
# plt.show()

# %%
# disp = ConfusionMatrixDisplay.from_predictions(y_true=gold_tags,
#                                                y_pred=pred_tags,
#                                                labels=categories,
#                                                xticks_rotation='vertical',
#                                                cmap=plt.cm.Blues)

# # %%
# categories = [
#     'ENTERTAIN', "ATTRIBUTE", "ENDORSE", "PRONOUNCE", "CONCUR", "COUNTER",
#     "DENY", "MONOGLOSS", "CITATION", "SOURCES", "ENDOPHORIC", "EXEMPLIFYING",
#     "EXPOSITORY", "JUSTIFYING", "SUMMATIVE", "COMPARATIVE", "TEXT_SEQUENCING",
#     "GOAL_ANNOUNCING", "_"
# ]

# main_categories = [
#     'ENTERTAIN', "ATTRIBUTE", "ENDORSE", "PRONOUNCE", "CONCUR", "COUNTER",
#     "DENY", "MONOGLOSS", "CITATION", "SOURCES", "ENDOPHORIC", "JUSTIFYING"
# ]
# main_categories = [
#     'ENTERTAIN', "ATTRIBUTE", "ENDORSE", "PRONOUNCE", "CONCUR", "COUNTER",
#     "DENY", "MONOGLOSS", "JUSTIFYING"
# ]

# a = classification_report(gold_tags, pred_tags, labels=main_categories)
# print(a)
# a = classification_report(gold_tags, pred_tags, labels=main_categories)
# print(a)
# a = classification_report(gold_tags, pred_tags, labels=main_categories)
# print(a)

# import pandas as pd

# y_true = pd.Series(gold_tags, name="Actual")
# y_pred = pd.Series(pred_tags, name="Predicted")
# df_confusion = pd.crosstab(y_true, y_pred)
# print(df_confusion)
# df_confusion.to_html('your_output_file_name.html')

# from sklearn.metrics import precision_recall_fscore_support

# precision_recall_fscore_support(gold_tags, pred_tags, labels=categories)

# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import cohen_kappa_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix

# # predict probabilities for test set
# yhat_probs = model.predict(testX, verbose=0)
# # predict crisp classes for test set
# yhat_classes = model.predict_classes(testX, verbose=0)
# # reduce to 1d array
# yhat_probs = yhat_probs[:, 0]
# yhat_classes = yhat_classes[:, 0]

# # accuracy: (tp + tn) / (p + n)
# accuracy = accuracy_score(
#     gold_tags,
#     pred_tags,
# )
# print('Accuracy: %f' % accuracy)
# # precision tp / (tp + fp)
# precision = precision_score(gold_tags, pred_tags, average='macro')
# print('Precision: %f' % precision)
# # recall: tp / (tp + fn)
# recall = recall_score(gold_tags, pred_tags, average='macro')
# print('Recall: %f' % recall)
# # f1: 2 tp / (2 tp + fp + fn)
# f1 = f1_score(gold_tags, pred_tags, average='macro')
# print('F1 score: %f' % f1)

# # kappa
# kappa = cohen_kappa_score(gold_tags, pred_tags, labels=categories)
# print('Cohens kappa: %f' % kappa)
# # ROC AUC

# auc = roc_auc_score(testy, yhat_probs)
# print('ROC AUC: %f' % auc)
# # confusion matrix
# matrix = confusion_matrix(testy, yhat_classes)
# print(matrix)
# # %%


def main(
    model: str = Arg(..., help="Model name or path"),
    data_path: Path = Arg(
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
        data_path,
        output=output,
        use_gpu=-1,
        gold_preproc=False,
        displacy_path=displacy_path,
        displacy_limit=25,
        silent=False,
    )


if __name__ == '__main__':
    typer.run(main)
