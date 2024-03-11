import spacy
from spacy.tokens import DocBin
from spacy.training.converters import conll_ner_to_docs

import pprint as pp

conll_data = '''He\tO\tO\tO\tO\tO
proceeded\tO\tB-MONOGLOSS\tO\tO\tO
,\tO\tO\tO\tO\tO
in\tO\tO\tO\tO\tO
response\tO\tO\tO\tO\tO
,\tO\tO\tO\tO\tO
to\tO\tO\tO\tO\tO
stop\tO\tO\tO\tO\tO
the\tO\tO\tO\tO\tO
conversation\tO\tO\tO\tO\tO
and\tO\tO\tO\tO\tO
to\tO\tO\tO\tO\tO
quiz\tO\tO\tO\tO\tO
me\tO\tO\tO\tO\tO
.\tO\tO\tO\tO\tO

demanding\tO\tB-MONOGLOSS\tO\tO\tO
that\tO\tO\tO\tO\tO
I\tO\tO\tO\tO\tO
name\tO\tB-MONOGLOSS\tO\tO\tO
six\tO\tO\tO\tO\tO
starting\tO\tO\tO\tO\tO
offensive\tO\tO\tO\tO\tO
players\tO\tO\tO\tO\tO
for\tO\tO\tO\tO\tO
the\tO\tO\tO\tO\tO
Notre\tO\tO\tO\tO\tO
Dame\tO\tO\tO\tO\tO
2006\tO\tO\tO\tO\tO
team\tO\tO\tO\tO\tO
.\tO\tO\tO\tO\tO\n


-DOCSTART- -X- O O\n
Two\tO\tO\tO\tO\tO
weeks\tO\tO\tO\tO\tO
later\tO\tO\tO\tO\tO
when\tO\tO\tO\tO\tO
back\tO\tO\tO\tO\tO
at\tO\tO\tO\tO\tO
school\tO\tO\tO\tO\tO
in\tO\tO\tO\tO\tO
Ann\tO\tO\tO\tO\tO
Arbor\tO\tO\tO\tO\tO
.\tO\tO\tO\tO\tO

I\tO\tO\tO\tO\tO
mentioned\tO\tB-MONOGLOSS\tO\tO\tO
to\tO\tO\tO\tO\tO
a\tO\tO\tO\tO\tO
male\tO\tO\tO\tO\tO
colleague\tO\tO\tO\tO\tO
of\tO\tO\tO\tO\tO
mine\tO\tO\tO\tO\tO
that\tO\tO\tO\tO\tO
I\tO\tO\tO\tO\tO
had\tO\tB-MONOGLOSS\tO\tO\tO
been\tO\tI-MONOGLOSS\tO\tO\tO
invited\tO\tI-MONOGLOSS\tO\tO\tO
to\tO\tO\tO\tO\tO
a\tO\tO\tO\tO\tO
"\tO\tO\tO\tO\tO
gays\tO\tO\tO\tO\tO
only\tO\tO\tO\tO\tO
gathering\tO\tO\tO\tO\tO
"\tO\tO\tO\tO\tO
for\tO\tO\tO\tO\tO
new\tO\tO\tO\tO\tO
students\tO\tO\tO\tO\tO
the\tO\tO\tO\tO\tO
following\tO\tO\tO\tO\tO
week\tO\tO\tO\tO\tO
,\tO\tO\tO\tO\tO
though\tO\tB-COUNTER\tO\tO\tO
I\tO\tI-COUNTER\tO\tO\tO
am\tB-MONOGLOSS\tI-COUNTER\tO\tO\tO
heterosexual\tO\tI-COUNTER\tO\tO\tO
.\tO\tO\tO\tO\tO
'''

doc_delimiter = "-DOCSTART- -X- O O\n"

num_levels = 3
docs = conll_data.split(doc_delimiter)  #separate into sents
len(docs)
iob_per_level = []
for level in range(num_levels):
    doc_list = []
    for doc in docs:  #iterate each chunk
        # print(doc)
        sent_list = []
        for sent in doc.split("\n\n"):
            print(sent)
            tokens = [t for t in sent.strip().split("\n") if t]  #tokens
            token_list = []
            for token in tokens:  #iterate tokens
                annot = token.split("\t")  #list of annotations
                # First element is always the token text
                text = annot[0]
                # text = text.replace("#", "_") #tested whether "#" was doing the trick
                # subsequent layers are relevant annotations
                label = annot[level + 1]

                # "text label" as format
                _token = " ".join([text, label])
                token_list.append(_token)
            sent_list.append("\n".join(token_list))
        doc_list.append("\n\n".join(sent_list))
    annotations = doc_delimiter.join(doc_list)
    iob_per_level.append(annotations)

pp.pprint(iob_per_level)

# We then copy all the entities from doc.ents into
# doc.spans later on. But first, let's have a "canonical" docs
# to copy into
# conll_ner_to_docs internally identifies whether sentence segmentation is done
docs_per_level = [list(conll_ner_to_docs(iob)) for iob in iob_per_level]

nlp = spacy.blank("en")
doc_bin = DocBin(docs=docs_per_level)
new_docs = list(doc_bin.get_docs(nlp.vocab))
docs_with_spans: List[Doc] = []

docs = [list(conll_ner_to_docs(iob))[0] for iob in iob_per_level]

nlp = spacy.blank("en")
doc_bin = DocBin(docs=docs)
new_docs = list(doc_bin.get_docs(nlp.vocab))
len(new_docs)
for docs in zip(*new_docs):
    for doc in docs:
        print(type(doc))
        print(doc.ents)
    assert new_docs[0].vocab == doc.vocab

docs_per_level = [list(conll_ner_to_docs(iob)) for iob in iob_per_level]
flat_docs = []
len(docs)

nlp = spacy.blank("en")

new_docs = []
for doc in docs_per_level:
    doc_bin = DocBin(docs=doc)
    new_doc = list(doc_bin.get_docs(nlp.vocab))
    new_docs.append(new_doc)

len(new_docs)
for docs in zip(*new_docs):
    for doc in docs:
        print(type(doc))
        print(doc.ents)
    assert new_docs[0].vocab == doc.vocab