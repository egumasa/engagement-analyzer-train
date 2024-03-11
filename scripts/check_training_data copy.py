import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")

training = DocBin().from_disk(
    "data/engagement_spl_test.spacy")

docs = list(training.get_docs(nlp.vocab))
print(len(docs))

for doc in docs:
    # print(doc)
    spanG = doc.spans['sc']
    # print(spanG)

    for x in spanG:

        if len(x) > 15:
            print(f"{x.label_}\tSPAN: {x.text}")


    for s in doc.sents:
        print(s)
        for t in doc:
            print(t)

dev = DocBin().from_disk(
    "data/engagement_spl_test.spacy")

docs = list(dev.get_docs(nlp.vocab))
print(len(docs))
