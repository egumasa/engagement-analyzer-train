import re
import spacy_streamlit
from collections import Counter

import spacy


from spacy.tokens import Doc
from spacy.cli._util import import_code

from utils.visualize import visualize_spans
from utils.util import preprocess, delete_overlapping_span, cleanup_justify

from resources.text_list import TEXT_LIST
from resources.template_list import TPL_SPAN, TPL_SPAN_SLICE, TPL_SPAN_START
from resources.colors import COLORS_1

from pipeline.post_processors import simple_table, const_table, ngrammar
import pandas as pd

# from pipeline.custom_functions import custom_functions
SPAN_ATTRS = ["text", "label_", "start", "end"]


# spacy.prefer_gpu()

def load_model(spacy_model):
    # source = spacy.blank("en")
    nlp = spacy.load(spacy_model)  # , vocab=nlp_to_copy.vocab
    nlp.add_pipe('sentencizer')
    return (nlp)

# source = spacy.blank("en")


import_code("pipeline/custom_functions.py")

nlp = spacy.load("en_engagement_three_RoBERTa_base_LSTM384")

doc = nlp(preprocess(TEXT_LIST[0]))

cleanup_justify(doc, doc.spans["sc"])
delete_overlapping_span(doc.spans['sc'])

data, cols = const_table(doc, spans_key='sc', attrs=SPAN_ATTRS)
seq = [s for s in doc.spans["sc"]]
span_ngrams = ngrammar(seq=seq, n=3)

df = pd.DataFrame(data, columns=cols)

constant_value = 42
new_col = pd.Series([constant_value] * df.shape[0], name='new_col')

doclen = len(doc)
doc_len = pd.Series([doclen] * df.shape[0], name='nwords')

df.insert(0, "new", new_col, True)
df.insert(1, "nwords", doc_len, True)

df.to_csv("results/test.csv")

st.subheader("Label counts & Diagnostic confidence score summary")
counts = df['label_'].value_counts()
label_counts = df.groupby('label_').agg({"label_": 'count',
                                         "Conf. score": ['median', 'min', 'max']}).round(4)

st.dataframe(label_counts)

st.subheader("Engagement label by grammatical function")
label_dep = pd.crosstab(df['span dep'], df['label_'])
st.dataframe(label_dep)

st.subheader('Quantitative results')
st.markdown(f"Shannon's index: {dv.alpha.shannon(counts, base=2): .3f}")
st.markdown(f"Simpson's e index: {dv.alpha.simpson_e(counts): .3f}")
