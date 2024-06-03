from pathlib import Path
from typing import List

import spacy
# from spacy.lang.en import English
import typer
from spacy.tokens import Doc, DocBin, SpanGroup, Span
from spacy.training.converters import conll_ner_to_docs
from wasabi import msg

DOC_DELIMITER = "-DOCSTART- -X- O O\n"


def parse_genia(data: str,
                span_key: str,
                num_levels: int = 4,
                doc_delimiter: str = DOC_DELIMITER) -> List[Doc]:
    """Parse GENIA dataset into spaCy docs

    Our strategy here is to reuse the conll -> ner method from
    spaCy and re-apply that n times. We don't want to write our
    own ConLL/IOB parser.

    Parameters
    ----------
    data: str
        The raw string input as read from the IOB file
    num_levels: int, default is 4
        Represents how many times a label has been nested. In
        GENIA, a label was nested four times at maximum.

    Returns
    -------
    List[Doc]
    """

    docs = data.split("\n\n")  #separate into sents
    iob_per_level = []
    for level in range(num_levels):
        doc_list = []
        for doc in docs:  #iterate each chunk
            tokens = [t for t in doc.split("\n") if t]  #tokens
            token_list = []
            for token in tokens:  #iterate tokens
                annot = token.split("\t")  #list of annotations
                # First element is always the token text
                text = annot[0]
                # subsequent layers are relevant annotations
                label = annot[level + 1]

                # "text label" as format
                _token = " ".join([text, label])
                token_list.append(_token)
            doc_list.append("\n".join(token_list))
        annotations = doc_delimiter.join(doc_list)
        iob_per_level.append(annotations)

    # We then copy all the entities from doc.ents into
    # doc.spans later on. But first, let's have a "canonical" docs
    # to copy into
    # conll_ner_to_docs internally identifies whether sentence segmentation is done
    docs_per_level = [list(conll_ner_to_docs(iob)) for iob in iob_per_level]
    docs_with_spans: List[Doc] = []

    for docs in zip(*docs_per_level):
        spans = [ent for doc in docs for ent in doc.ents]
        doc = docs[0]
        group = SpanGroup(doc, name=span_key, spans=spans)
        doc.spans[span_key] = group
        docs_with_spans.append(doc)

    return docs_with_spans


def parse_engagement_v2(data: str,
                        span_key: str,
                        num_levels: int = 4,
                        doc_delimiter: str = DOC_DELIMITER,
                        nlp=None) -> List[Doc]:
    """Parse ENGAGEMENT dataset into spaCy docs

    Our strategy here is to reuse the conll -> ner method from
    spaCy and re-apply that n times. We don't want to write our
    own ConLL/IOB parser.

    Parameters
    ----------
    data: str
        The raw string input as read from the IOB file
    num_levels: int, default is 4
        Represents how many times a label has been nested. In
        GENIA, a label was nested four times at maximum.

    Returns
    -------
    List[Doc]
    """
    # docs = data.split("\n\n") #separate into sents
    docs = data.split(doc_delimiter)

    iob_per_level = []
    for level in range(num_levels):
        doc_list = []
        for doc in docs:  #iterate each chunk
            # print(doc)
            sent_list = []
            for sent in doc.split("\n\n"):
                tokens = [t for t in sent.split("\n") if t]  #tokens
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

        # We then copy all the entities from doc.ents into
        # doc.spans later on. But first, let's have a "canonical" docs
        # to copy into
        # conll_ner_to_docs internally identifies whether sentence segmentation is done
    docs_per_level = [list(conll_ner_to_docs(iob)) for iob in iob_per_level]

    docs_with_spans: List[Doc] = []

    for docs in zip(*docs_per_level):
        for doc in docs:
            print(type(doc))
        spans = [ent for doc in docs for ent in doc.ents]
        # print(spans)
        # print([span.label_ for span in spans])
        doc = docs[0]
        group = SpanGroup(doc, name=span_key, spans=spans)
        doc.spans[span_key] = group
        docs_with_spans.append(doc)

    return docs_with_spans


def parse_engagement_v3(data: str,
                        span_key: str,
                        num_levels: int = 4,
                        doc_delimiter: str = DOC_DELIMITER,
                        nlp=None) -> List[Doc]:
    """Parse ENGAGEMENT dataset into spaCy docs

    Our strategy here is to reuse the conll -> ner method from
    spaCy and re-apply that n times. We don't want to write our
    own ConLL/IOB parser.

    Parameters
    ----------
    data: str
        The raw string input as read from the IOB file
    num_levels: int, default is 4
        Represents how many times a label has been nested. In
        GENIA, a label was nested four times at maximum.

    Returns
    -------
    List[Doc]
    """
    # docs = data.split("\n\n") #separate into sents
    docs = data.split(doc_delimiter)

    iob_per_level = []
    for level in range(num_levels):
        doc_list = []
        for doc in docs:  #iterate each chunk
            # print(doc)
            sent_list = []
            for sent in doc.split("\n\n"):
                tokens = [t for t in sent.split("\n") if t]  #tokens
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

        # We then copy all the entities from doc.ents into
        # doc.spans later on. But first, let's have a "canonical" docs
        # to copy into
        # conll_ner_to_docs internally identifies whether sentence segmentation is done
    docs_per_level = [list(conll_ner_to_docs(iob)) for iob in iob_per_level]

    nlp = spacy.blank("en")

    # doc_bin = DocBin(docs=docs_per_level)
    # new_docs = list(doc_bin.get_docs(nlp.vocab))
    new_docs = []
    for doc in docs_per_level:
        doc_bin = DocBin(docs=doc)
        new_doc = list(doc_bin.get_docs(nlp.vocab))
        new_docs.append(new_doc)

    docs_with_spans: List[Doc] = []

    for docs in zip(*new_docs):
        # print(type(docs))
        spans = [ent for doc in docs for ent in doc.ents]
        # print(spans)
        # print([span.label_ for span in spans])
        doc = docs[0]

        group = []
        for span in spans:
            group.append(Span(doc, span.start, span.end, span.label_))

        ## group = SpanGroup(doc, name=span_key, spans=spans) #This being the original
        doc.spans[span_key] = group
        docs_with_spans.append(doc)

    return docs_with_spans


def main(input_path: Path, output_path: Path, span_key: str):
    nlp = spacy.blank("en")
    msg.good(f"Processing Engagement dataset ")
    with input_path.open("r", encoding="utf-8") as f:
        data = f.read()

    docs = parse_engagement_v3(data, span_key=span_key, num_levels=3, nlp=nlp)
    # docs = parse_engagement_v3(data, span_key=span_key, num_levels=3, nlp=nlp)
    # docs = parse_genia(data, span_key=span_key)
    doc_bin = DocBin(docs=docs)
    doc_bin.to_disk(output_path)

    msg.good(f"Processing Engagement dataset done")


if __name__ == "__main__":
    typer.run(main)
