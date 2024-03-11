import re
import spacy_streamlit
from collections import Counter

import spacy
import streamlit as st

# try:
#     from .scripts.custom_functions import build_mean_max_reducer1, build_mean_max_reducer2, build_mean_max_reducer3
# except ImportError:
#     from pipeline.custom_functions import build_mean_max_reducer1, build_mean_max_reducer2, build_mean_max_reducer3
from spacy.tokens import Doc
from spacy.cli._util import import_code

from utils.visualize import visualize_spans
from utils.util import preprocess, delete_overlapping_span, cleanup_justify
from resources.text_list import TEXT_LIST
from resources.template_list import TPL_SPAN, TPL_SPAN_SLICE, TPL_SPAN_START
from resources.colors import COLORS_1

import_code("Engagement-Analyzer/pipeline/custom_functions.py")
st.set_page_config(page_title='Engagement model comparaer', layout="wide")

# spacy.prefer_gpu()

MODEL_LIST = [
    'en_engagement_three_RoBERTa_base_LSTM384-0.9.2/en_engagement_three_RoBERTa_base_LSTM384/en_engagement_three_RoBERTa_base_LSTM384-0.9.2',
    'en_engagement_three_RoBERTa_acad3_db-0.9.2/en_engagement_three_RoBERTa_acad3_db/en_engagement_three_RoBERTa_acad3_db-0.9.2',
    'silver-sweep-34/model-best',
    'expert-sweep-4/model-best',
    'confused-sweep-6/model-best',
    'warm-sweep-20/model-best',
    "en_engagement_three_RoBERTa_base-1.10.0/en_engagement_three_RoBERTa_base/en_engagement_three_RoBERTa_base-1.10.0",
    "en_engagement_three_RoBERTa_acad_db-1.10.0/en_engagement_three_RoBERTa_acad_db/en_engagement_three_RoBERTa_acad_db-1.10.0",
    "en_engagement_para_RoBERTa_acad_db3-0.9.0/en_engagement_para_RoBERTa_acad_db3/en_engagement_para_RoBERTa_acad_db3-0.9.0",
    "en_engagement_para_RoBERTa_acad_LSTM2-0.9.0/en_engagement_para_RoBERTa_acad_LSTM2/en_engagement_para_RoBERTa_acad_LSTM2-0.9.0",
    "en_engagement_three_RoBERTa_acad_db3-0.9.1/en_engagement_three_RoBERTa_acad_db3/en_engagement_three_RoBERTa_acad_db3-0.9.1",
    "en_engagement_three_RoBERTa_acad_LSTM2-0.9.1/en_engagement_three_RoBERTa_acad_LSTM2/en_engagement_three_RoBERTa_acad_LSTM2-0.9.1",
    "en_engagement_three_RoBERTa_acad_db3-0.9.2/en_engagement_three_RoBERTa_acad_db3/en_engagement_three_RoBERTa_acad_db3-0.9.2",
    'en_engagement_spl_RoBERTa_acad_db-0.7.4/en_engagement_spl_RoBERTa_acad_db/en_engagement_spl_RoBERTa_acad_db-0.7.4',
    'en_engagement_spl_RoBERTa_acad_db3-0.9.0/en_engagement_spl_RoBERTa_acad_db3/en_engagement_spl_RoBERTa_acad_db3-0.9.0',
    'en_engagement_spl_RoBERTa_acad_LSTM-0.7.2/en_engagement_spl_RoBERTa_acad_LSTM/en_engagement_spl_RoBERTa_acad_LSTM-0.7.2',
    'en_engagement_spl_RoBERTa_acad_512',
    'en_engagement_spl_RoBERTa_acad',
    'en_engagement_spl_RoBERTa_exp-0.6.5/en_engagement_spl_RoBERTa_exp/en_engagement_spl_RoBERTa_exp-0.6.5',
    # 'en_engagement_spl_RoBERTa_acad-0.3.4.1221/en_engagement_spl_RoBERTa_acad/en_engagement_spl_RoBERTa_acad-0.3.4.1221',
    # 'en_engagement_spl_RoBERTa_acad-0.2.2.1228/en_engagement_spl_RoBERTa_acad/en_engagement_spl_RoBERTa_acad-0.2.2.1228',
    # 'en_engagement_spl_RoBERTa_acad-0.2.1.1228/en_engagement_spl_RoBERTa_acad/en_engagement_spl_RoBERTa_acad-0.2.1.1228',
    # 'en_engagement_spl_RoBERTa_acad-0.2.2.1220/en_engagement_spl_RoBERTa_acad/en_engagement_spl_RoBERTa_acad-0.2.2.1220',
    # 'en_engagement_spl_RoBERTa2-0.2.2.1210/en_engagement_spl_RoBERTa2/en_engagement_spl_RoBERTa2-0.2.2.1210',
    # 'en_engagement_spl_RoBERTa-0.2.2.1210/en_engagement_spl_RoBERTa/en_engagement_spl_RoBERTa-0.2.2.1210',
    # 'en_engagement_spl_RoBERTa_acad_max1_do02',
    # 'en_engagement_spl_RoBERTa2-0.2.2.1210/en_engagement_spl_RoBERTa2/en_engagement_spl_RoBERTa2-0.2.2.1210',
    # 'en_engagement_spl_RoBERTa_acad-0.2.3.1210/en_engagement_spl_RoBERTa_acad/en_engagement_spl_RoBERTa_acad-0.2.3.1210',
    # 'en_engagement_spl_RoBERTa_acad_max1_do02',
    # 'en_engagement_spl_RoBERTa_sqbatch_RAdam-20221202_0.1.5/en_engagement_spl_RoBERTa_sqbatch_RAdam/en_engagement_spl_RoBERTa_sqbatch_RAdam-20221202_0.1.5',
    # 'en_engagement_spl_RoBERTa_context_flz-20221130_0.1.4/en_engagement_spl_RoBERTa_context_flz/en_engagement_spl_RoBERTa_context_flz-20221130_0.1.4',
    # 'en_engagement_spl_RoBERTa_cx_max1_do2-20221202_0.1.5/en_engagement_spl_RoBERTa_cx_max1_do2/en_engagement_spl_RoBERTa_cx_max1_do2-20221202_0.1.5',
    # 'en_engagement_spl_RoBERTa_context_flz-20221125_0.1.4/en_engagement_spl_RoBERTa_context_flz/en_engagement_spl_RoBERTa_context_flz-20221125_0.1.4',
    # 'en_engagement_RoBERTa_context_flz-20221125_0.1.4/en_engagement_RoBERTa_context_flz/en_engagement_RoBERTa_context_flz-20221125_0.1.4',
    # 'en_engagement_RoBERTa_context_flz-20221117_0.1.3/en_engagement_RoBERTa_context_flz/en_engagement_RoBERTa_context_flz-20221117_0.1.3',
    # 'en_engagement_spl_RoBERTa_acad_context_flz-20221117_0.1.3/en_engagement_spl_RoBERTa_acad_context_flz/en_engagement_spl_RoBERTa_acad_context_flz-20221117_0.1.3',
    # 'en_engagement_RoBERTa_context_flz-Batch2_0.1.1/en_engagement_RoBERTa_context_flz/en_engagement_RoBERTa_context_flz-Batch2_0.1.1',
    # 'en_engagement_RoBERTa_context_flz-20221113_0.1.3/en_engagement_RoBERTa_context_flz/en_engagement_RoBERTa_context_flz-20221113_0.1.3',
    # 'en_engagement_RoBERTa_context_flz-20221113_0.1.1/en_engagement_RoBERTa_context_flz/en_engagement_RoBERTa_context_flz-20221113_0.1.1',
    # 'en_engagement_RoBERTa-0.0.2/en_engagement_RoBERTa/en_engagement_RoBERTa-0.0.2',
    # 'en_engagement_RoBERTa_combined-Batch2Eng_0.2/en_engagement_RoBERTa_combined/en_engagement_RoBERTa_combined-Batch2Eng_0.2',
    # 'en_engagement_RoBERTa_acad-0.2.1/en_engagement_RoBERTa_acad/en_engagement_RoBERTa_acad-0.2.1',
    # # 'en_engagement_BERT-0.0.2/en_engagement_BERT/en_engagement_BERT-0.0.2',
    # # 'en_engagement_BERT_acad-0.0.2/en_engagement_BERT_acad/en_engagement_BERT_acad-0.0.2',
    # # 'en_engagement_RoBERTa_acad-0.0.2/en_engagement_RoBERTa_acad/en_engagement_RoBERTa_acad-0.0.2',
    # 'en_engagement_RoBERTa-0.0.1/en_engagement_RoBERTa/en_engagement_RoBERTa-0.0.1',
    # # ' en_engagement_RoBERTa_sent-0.0.1_null/en_engagement_RoBERTa_sent/en_engagement_RoBERTa_sent-0.0.1_null',
    # # 'en_engagement_RoBERTa_combined-0.0.1/en_engagement_RoBERTa_combined/en_engagement_RoBERTa_combined-0.0.1',
    # 'en_engagement_RoBERTa-ME_AtoE/en_engagement_RoBERTa/en_engagement_RoBERTa-ME_AtoE',
    # 'en_engagement_RoBERTa-AtoI_0.0.3/en_engagement_RoBERTa/en_engagement_RoBERTa-AtoI_0.0.3',
    # 'en_engagement_RoBERTa-AtoI_0.0.3/en_engagement_RoBERTa/en_engagement_RoBERTa-AtoI_0.0.2'
]

multicol = st.checkbox("Compare two models", value=True, key=None, help=None)

model1 = st.selectbox('Select model option 1', MODEL_LIST, index=0)
model2 = st.selectbox('Select model option 2', MODEL_LIST, index=1)

if '/' in model1:
    model1 = "packages/" + model1

if '/' in model2:
    model2 = "packages/" + model2


@st.cache(allow_output_mutation=True)
def load_model(spacy_model):
    # source = spacy.blank("en")
    nlp = spacy.load(spacy_model) #, vocab=nlp_to_copy.vocab
    nlp.add_pipe('sentencizer')
    return (nlp)

# source = spacy.blank("en")
nlp = load_model(model1)

if multicol:
    nlp2 = load_model(model2)


text = st.selectbox('select sent to debug', TEXT_LIST)

input_text = st.text_area("", height=200)

# Dependency parsing
st.header("Text", "text")
if len(input_text.split(" ")) > 1:
    doc = nlp(preprocess(input_text))
    if multicol:
        doc2 = nlp2(preprocess(input_text))
    # st.markdown("> " + input_text)
else:
    doc = nlp(preprocess(text))
    if multicol:
        doc2 = nlp2(preprocess(text))
    # st.markdown("> " + text)

clearjustify = st.checkbox(
    "Clear problematic JUSTIFYING spans", value=True, key=None, help=None)

delete_overlaps = st.checkbox(
    "Delete overlaps", value=True, key=None, help=None)

# combine = st.checkbox(
#     "Combine", value=False, key=None, help=None)

# import copy
# def combine_spangroups(doc1, doc2):
#     # new_doc = Doc.from_docs([doc1, doc2], ensure_whitespace=True)
#     new_doc = copy.deepcopy(doc1)
#     # type()
#     new_doc.spans['sc'].extend(doc2.spans['sc'])

#     return new_doc


# if combine:
#     new_doc = combine_spangroups(doc, doc2)
#     visualize_spans(new_doc,
#                     spans_key="sc",
#                     title='Combined spans:',
#                     displacy_options={
#                         'template': {
#                               "span": TPL_SPAN,
#                             'slice': TPL_SPAN_SLICE,
#                             'start': TPL_SPAN_START,
#                         },
#                         "colors": COLORS_1,
#                     },
#                     simple=False)

if clearjustify:
    cleanup_justify(doc, doc.spans['sc'])

if delete_overlaps:
    delete_overlapping_span(doc.spans['sc'])
    if multicol:
        delete_overlapping_span(doc2.spans['sc'])

if not multicol:
    visualize_spans(doc,
                    spans_key="sc",
                    title='Engagement Span Anotations 1',
                    displacy_options={
                        'template': {
                              "span": TPL_SPAN,
                            'slice': TPL_SPAN_SLICE,
                            'start': TPL_SPAN_START,
                        },
                        "colors": COLORS_1,
                    },
                    simple=False)


else:
    col1, col2 = st.columns(2)

    with col1:
        visualize_spans(doc,
                        spans_key="sc",
                        title='Engagement Span Anotations 1',
                        displacy_options={
                            'template': {
                                "span": TPL_SPAN,
                                'slice': TPL_SPAN_SLICE,
                                'start': TPL_SPAN_START,
                            },
                            "colors": COLORS_1,
                        },
                        simple=False)

    with col2:
        visualize_spans(doc2,
                        spans_key="sc",
                        title='Engagement Span Anotations 2',
                        displacy_options={
                            'template': {
                                "span": TPL_SPAN,
                                'slice': TPL_SPAN_SLICE,
                                'start': TPL_SPAN_START,
                            },
                            "colors": COLORS_1,
                        },
                        simple=False)
