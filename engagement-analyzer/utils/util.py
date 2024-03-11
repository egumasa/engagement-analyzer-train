
import re
from collections import Counter
from spacy.tokens import SpanGroup


def preprocess(text):
    text = re.sub("\n\n", ' &&&&&&&&#&#&#&#&', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\s+', " ", text)
    text = re.sub('&&&&&&&&#&#&#&#&', '\n\n', text)
    return text


def del_spans(span_sc, indexes: list):

    indexes.sort(reverse=True) # reversing allows the deletion from the last, keeping the original index

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
            justifies[span.root] = {"span": span,
                                    "head": span.root.head,
                                    "start": span.start,
                                    "end": span.end,
                                    "del": False,
                                    "dependent": False,
                                    "span_idx": idx}
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

    to_delete = [info['span_idx']
                 for spanroot, info in justifies.items() if info['del']]

    to_delete_span = [info['span']
                      for spanroot, info in justifies.items() if info['del']]

    # print(to_delete)
    # print(to_delete_span)

    del_spans(span_sc, to_delete)

    span_grp = SpanGroup(doc, spans=new_spans)
    span_sc.extend(span_grp)

    # print(justifies)
