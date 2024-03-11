#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# This code is adapted from spacy-streamlit package by explosion
# https://github.com/explosion/spacy-streamlit/blob/master/spacy_streamlit/__init__.py
#

from typing import List, Sequence, Tuple, Optional, Dict, Union, Callable
import streamlit as st
import spacy
from spacy.language import Language
from spacy import displacy
import pandas as pd


import streamlit as st
from spacy_streamlit import visualize_spans
from spacy_streamlit.util import load_model, process_text, get_svg, get_html, LOGO

from pipeline.post_processors import simple_table, const_table, ngrammar
from skbio import diversity as dv

SPACY_VERSION = tuple(map(int, spacy.__version__.split(".")))

# fmt: off
# SPAN_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]
SPAN_ATTRS = ["text", "label_", "start", "end",]

def visualize_spans(
    doc: Union[spacy.tokens.Doc, Dict[str, str]],
    *,
    spans_key: str = "sc",
    attrs: List[str] = SPAN_ATTRS,
    show_table: bool = True,
    title: Optional[str] = "Spans",
    manual: bool = False,
    displacy_options: Optional[Dict] = None,
    simple: bool = True,
):
    """
    Visualizer for spans.
    doc (Doc, Dict): The document to visualize.
    spans_key (str): Which spans key to render spans from. Default is "sc".
    attrs (list):  The attributes on the entity Span to be labeled. Attributes are displayed only when the show_table
    argument is True.
    show_table (bool): Flag signifying whether to show a table with accompanying span attributes.
    title (str): The title displayed at the top of the Spans visualization.
    manual (bool): Flag signifying whether the doc argument is a Doc object or a List of Dicts containing span information.
    displacy_options (Dict): Dictionary of options to be passed to the displacy render method for generating the HTML to be rendered.
      See https://spacy.io/api/top-level#displacy_options-span
    """
    if SPACY_VERSION < (3, 3, 0):
        raise ValueError(
            f"'visualize_spans' requires spacy>=3.3.0. You have spacy=={spacy.__version__}"
        )
    if not displacy_options:
        displacy_options = dict()
    displacy_options["spans_key"] = spans_key

    if title:
        st.header(title)

    if manual:
        if show_table:
            st.warning(
                "When the parameter 'manual' is set to True, the parameter 'show_table' must be set to False."
            )
        if not isinstance(doc, dict):
            st.warning(
                "When the parameter 'manual' is set to True, the parameter 'doc' must be of type 'Dict', not 'spacy.tokens.Doc'."
            )
    html = displacy.render(
        doc,
        style="span",
        options=displacy_options,
        manual=manual,
    )
    st.write(f"{get_html(html)}", unsafe_allow_html=True)

    if show_table:
        # data = [
        #     [str(getattr(span, attr)) for attr in attrs] + [str(score)]
        #     for span, score in zip(doc.spans[spans_key], doc.spans[spans_key].attrs['scores'])
        # ]
        if simple:
            data, cols = simple_table(doc, spans_key='sc', attrs=attrs)
        else:
            data, cols = const_table(doc, spans_key='sc', attrs=attrs)

        seq = [s for s in doc.spans[spans_key]]

        span_ngrams = ngrammar(seq=seq, n = 3)
        st.code(span_ngrams)


        if data:
            df = pd.DataFrame(data, columns=cols)
            st.dataframe(df.style.highlight_between(subset= 'Conf. score', right = .7))

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
            # st.markdown(str(dv.alpha_diversity(metric = "shannon", counts=counts, ids = ['ENTERTAIN', 'ATTRIBUTE', 'CITATION', 'COUNTER', 'DENY', 'ENDORSE', 'PRONOUNCE', 'CONCUR', 'MONOGLOSS', 'SOURCES', 'JUSTIFYING'])))
            # print(dv.get_alpha_diversity_metrics())

