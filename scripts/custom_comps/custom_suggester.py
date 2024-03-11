
## Original
import spacy 

from typing import List, Optional, Iterable, cast
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d
from spacy.pipeline.spancat import Suggester
from spacy.tokens import Doc
from spacy.util import registry
from typing import List, Optional, cast
import numpy
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d

from spacy.tokens.token import Token

def merge_suggestions(suggestions: List[Ragged], ops: Optional[Ops] = None) -> Ragged:
    '''
    This function is taken from: 
    https://github.com/explosion/spacy-experimental/blob/master/spacy_experimental/span_suggesters/merge_suggesters.py
    '''
    if ops is None:
        ops = get_current_ops()

    spans = []
    lengths = []

    if len(suggestions) == 0:
        lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
        return Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

    len_docs = len(suggestions[0])
    assert all(len_docs == len(x) for x in suggestions)

    for i in range(len_docs):
        combined = ops.xp.vstack([s[i].data for s in suggestions if len(s[i].data) > 0])
        uniqued = numpy.unique(ops.to_numpy(combined), axis=0)
        spans.append(ops.asarray(uniqued))
        lengths.append(uniqued.shape[0])

    lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
    if len(spans) > 0:
        output = Ragged(ops.xp.vstack(spans), lengths_array)
    else:
        output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

    return output

def build_ngram_subtree_suggester(sizes: List[int]) -> Suggester:
    """Suggest ngrams and subtrees. Requires annotations from the DependencyParser"""

    ngram_suggester = registry.misc.get("spacy.ngram_suggester.v1")(sizes)

    def ngram_subtree_suggester(
        docs: Iterable[Doc], *, ops: Optional[Ops] = None
    ) -> Ragged:
        ngram_suggestions = ngram_suggester(docs, ops=ops)
        subtree_suggestions = subtree_suggester(docs, ops=ops)
        return merge_suggestions([ngram_suggestions, subtree_suggestions], ops=ops)

    return ngram_subtree_suggester


@registry.misc("subtree_suggester.v1")
def build_subtree_suggester() -> Suggester:
    """Suggest subtrees. Requires annotations from the DependencyParser"""
    return subtree_suggester

def subtree_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
    if ops is None:
        ops = get_current_ops()

    spans = []
    spandoc = []
    lengths = []

    for doc in docs:
        cache = set()
        length = 0

        for token in doc:
            if (token.left_edge.i, token.i + 1) not in cache:
                spans.append((token.left_edge.i, token.i + 1))
                spandoc.append(doc[token.left_edge.i: token.i + 1])
                cache.add((token.left_edge.i, token.i + 1))
                length += 1
            if (token.i, token.right_edge.i + 1) not in cache:
                spans.append((token.i, token.right_edge.i + 1))
                spandoc.append(doc[token.i: token.right_edge.i + 1])
                cache.add((token.i, token.right_edge.i + 1))
                length += 1
            if (token.left_edge.i, token.right_edge.i + 1) not in cache:
                spans.append((token.left_edge.i, token.right_edge.i + 1))
                spandoc.append(doc[token.left_edge.i: token.right_edge.i + 1])
                cache.add((token.left_edge.i, token.right_edge.i + 1))
                length += 1

        lengths.append(length)

    print(spandoc)
    lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
    if len(spans) > 0:
        output = Ragged(ops.asarray(spans, dtype="i"), lengths_array)
    else:
        output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

    return output



nlp = spacy.load("en_core_web_trf")
text = "My guess is that you would probably not believe in this approach yet."
doc = nlp(text)

sgstr = build_ngram_subtree_suggester([1,2,3,4,5,6,7,8,9,10,11,12])

subtree_suggester([doc])