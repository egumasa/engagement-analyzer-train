
from typing import List, Sequence, Tuple, Optional, Dict, Union, Callable
import pandas as pd
import spacy
from spacy.language import Language

SPAN_ATTRS = ["text", "label_", "start", "end"]


def simple_table(doc: Union[spacy.tokens.Doc, Dict[str, str]],
                 spans_key: str = "sc",
                 attrs: List[str] = SPAN_ATTRS):
    columns = attrs + ["Conf. score"]
    data = [
        [str(getattr(span, attr))
         for attr in attrs] + [score]  # [f'{score:.5f}']
        for span, score in zip(doc.spans[spans_key], doc.spans[spans_key].attrs['scores'])
    ]
    return data, columns


# def span_info_aggregator()

def construction_classifier(doc, span):
    category = span.root.dep_
    spanroot = span.root

    ##
    span_t_dep_ = ["_".join([t.norm_, t.dep_]) for t in span]
    span_dep = [t.dep_ for t in span]
    span_token = [t.norm_ for t in span]
    span_tag = [t.tag_ for t in span]


    c_dep = [c.dep_ for c in spanroot.children]
    c_pos = [c.pos_ for c in spanroot.children]
    c_tag = [c.tag_ for c in spanroot.children]

    ## nesting classifiers
    if spanroot.dep_ == "conj":
        while spanroot.dep_ == 'conj':
            spanroot = spanroot.head
    if spanroot.dep_ == "poss":
        while spanroot.dep_ == 'poss':
            spanroot = spanroot.head


    ## Simple classifier
    if spanroot.dep_ in ['pcomp']:
        if str(spanroot.morph) in ["Aspect=Prog|Tense=Pres|VerbForm=Part"]:
            category = "Gerund"


    if spanroot.dep_ in ["pobj", "dobj", "obj", "iobj"]:
        category = "Object"
    if spanroot.dep_ in ["nsubj", "nsubjpass"]:
        category = "Subject"
    if spanroot.dep_ in ["cc"]:
        category = "Coordinating conjunction"

    if spanroot.dep_ in ["ROOT", "advcl"]:
        if "ccomp" in c_dep and "auxpass" in c_dep and ("it_nsubjpass" in span_t_dep_ or "it_nsubj" in span_t_dep_):
            category = "It is X that-clause"
        elif "nsubj" in c_dep and "acomp" in c_dep and ("it_nsubjpass" in span_t_dep_ or "it_nsubj" in span_t_dep_):
            category = "It is X that-clause"
        elif "nsubj" in c_dep and "oprd" in c_dep and ("it_nsubjpass" in span_t_dep_ or "it_nsubj" in span_t_dep_):
            category = "It is X that-clause"
        elif "nsubj" in c_dep and "it" in span_token and spanroot.pos_ == "VERB":
            category = "It VERB that-clause"
        elif "expl" in c_dep and "NOUN" in c_pos:
            category = "There is/are NOUN"
        elif spanroot.pos_ in ["AUX", 'VERB']:
            category = "Main verb"
        else:
            category = spanroot.dep_
    
    if spanroot.dep_ in ['attr']:
        c_head = [c.dep_ for c in spanroot.head.children]
        if "expl" in c_head and "no_det" in span_t_dep_:
            category = "There is/are no NOUN"

        
    # Modal verbs
    if spanroot.tag_ == "MD":
        category = "Modal auxiliary"
    # prep phrases
    if spanroot.dep_ in ['prep']:
        category = 'Prepositional Phrase'
    # adverbial phrases
    if spanroot.dep_ in ['advmod']:
        category = "Adverbial modifier"
        # adverbial phrases
    if spanroot.dep_ in ['acomp']:
        category = "Adjectival complement"

    # Preconjunctions
    if spanroot.dep_ in ['preconj']:
        category = "Conjunction"

    # Adverbial clauses
    ## Check the status of the adverbial clauses carefully
    if spanroot.dep_ in ['advcl', 'mark', 'acl']:
        if "mark" in span_dep:
            category = "Finite adverbial clause"
        if str(spanroot.morph) in ["Aspect=Prog|Tense=Pres|VerbForm=Part"] and "aux" not in c_dep:
            category = "Non-finite adv clause"
        # Check whether it has a subject or not
        # elif "nsubj" in [c.dep_ for c in spanroot.children]:
        #     category = "Adverbial clauses"
        # else:
        #     category = "Other advcl"
    
    if spanroot.dep_ in ['relcl', 'ccomp']:
        head = spanroot.head
        if ";" in [t.norm_ for t in head.children]:
            category = "Main verb"
        elif "nsubj" not in span_dep:
            category = "Dependent verb"

    if spanroot.dep_ in ['dep']:
        if spanroot.head.dep_ in ['ROOT', 'ccomp'] and spanroot.head.pos_ in ['AUX', 'VERB'] and spanroot.pos_ in ['AUX', 'VERB']:
            if spanroot.morph == spanroot.head.morph:
                category = "Main verb"
            else:
                category = "Dependent verb"




    if span.label_ == "CITATION":
        if "NNP" in span_tag or "NNPS" in span_tag:
            if span_dep[0] == 'punct' and span_dep[-1] == 'punct':
                category = "Parenthetical Citation"
            elif span_tag[0] in ["NNP", "NNPS"]:
                category = "Narrative Citation"
        else:
            category = "Other Citation"


    return category


def const_table(doc: Union[spacy.tokens.Doc, Dict[str, str]],
                spans_key: str = "sc",
                attrs: List[str] = SPAN_ATTRS):
    columns = attrs + ["Conf. score", "sent no.", "grammatical realization", 'span dep', "ner", 
                       "POS", 'span dep seq', "POS sequence", "head", "children", "morphology", ]
    data = []
    # data = span_info_aggregator(doc, columns)
    sentences = {s: i for i, s in enumerate(doc.sents)}

    for span, score in zip(doc.spans[spans_key], doc.spans[spans_key].attrs['scores']):

        span_info = []
        span_info.extend([str(getattr(span, attr)) for attr in attrs])

        span_info.append(score)
        span_info.append(sentences[span.sent])
        span_info.append(construction_classifier(doc, span))
        span_info.append(span.root.dep_)
        span_info.append(span.root.ent_type_)
        span_info.append(span.root.tag_)
        span_info.append("_".join([t.dep_ for t in span]))
        span_info.append("_".join([t.tag_ for t in span]))
        span_info.append(span.root.head.norm_)
        span_info.append("_".join([c.dep_ for c in span.root.children]))
        span_info.append(span.root.morph)
        data.append(span_info)

    return data, columns


def ngrammar(seq: list, n=2):
    result = []
    n_item = len(seq)
    for idx, item in enumerate(seq):
        if idx + n <= n_item:
            result.append(seq[idx: idx + n])
    return result
