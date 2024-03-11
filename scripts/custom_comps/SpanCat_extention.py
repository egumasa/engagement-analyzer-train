from typing import List, Tuple, cast
from thinc.api import Model, with_getitem, chain, list2ragged, Logistic
from thinc.api import Maxout, Linear, concatenate, glorot_uniform_init, PyTorchLSTM
from thinc.api import reduce_mean, reduce_max, reduce_first, reduce_last
from thinc.types import Ragged, Floats2d

from spacy.util import registry
from spacy.tokens import Doc
from spacy.ml.extract_spans import extract_spans

# @registry.layers("spacy.LinearLogistic.v1")
# def build_linear_logistic(nO=None, nI=None) -> Model[Floats2d, Floats2d]:
#     """An output layer for multi-label classification. It uses a linear layer
#     followed by a logistic activation.
#     """
#     return chain(Linear(nO=nO, nI=nI, init_W=glorot_uniform_init), Logistic())


@registry.layers("mean_max_reducer.v1.5")
def build_mean_max_reducer1(hidden_size: int,
                           dropout: float = 0.0) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    return chain(
        concatenate(
            cast(Model[Ragged, Floats2d], reduce_last()),
            cast(Model[Ragged, Floats2d], reduce_first()),
            reduce_mean(),
            reduce_max(),
        ),
        Maxout(nO=hidden_size, normalize=True, dropout=dropout),
    )


@registry.layers("mean_max_reducer.v2")
def build_mean_max_reducer2(hidden_size: int,
                            dropout: float = 0.0) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    return chain(
        concatenate(
            cast(Model[Ragged, Floats2d], reduce_last()),
            cast(Model[Ragged, Floats2d], reduce_first()),
            reduce_mean(),
            reduce_max(),
        ), Maxout(nO=hidden_size, normalize=True, dropout=dropout),
        Maxout(nO=hidden_size, normalize=True, dropout=dropout))


# @registry.layers("mean_max_reducer.v2")
# def build_mean_max_reducer2(hidden_size: int,
#                             depth: int) -> Model[Ragged, Floats2d]:
#     """Reduce sequences by concatenating their mean and max pooled vectors,
#     and then combine the concatenated vectors with a hidden layer.
#     """
#     return chain(
#         concatenate(
#             cast(Model[Ragged, Floats2d], reduce_last()),
#             cast(Model[Ragged, Floats2d], reduce_first()),
#             reduce_mean(),
#             reduce_max(),
#         ), Maxout(nO=hidden_size, normalize=True, dropout=0.0),
#         PyTorchLSTM(nO=64, nI=hidden_size, bi=True, depth=depth, dropout=0.2))


@registry.layers("mean_max_reducer.v3")
def build_mean_max_reducer3(hidden_size: int,
                            maxout_pieces: int = 3,
                            dropout: float = 0.0) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    hidden_size2 = int(hidden_size / 2)
    hidden_size3 = int(hidden_size / 2)
    return chain(
        concatenate(
            cast(Model[Ragged, Floats2d], reduce_last()),
            cast(Model[Ragged, Floats2d], reduce_first()),
            reduce_mean(),
            reduce_max(),
        ),
        Maxout(nO=hidden_size,
               nP=maxout_pieces,
               normalize=True,
               dropout=dropout),
        Maxout(nO=hidden_size2,
               nP=maxout_pieces,
               normalize=True,
               dropout=dropout),
        Maxout(nO=hidden_size3,
               nP=maxout_pieces,
               normalize=True,
               dropout=dropout))


@registry.layers("mean_max_reducer.v3.3")
def build_mean_max_reducer4(hidden_size: int,
                            depth: int) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    hidden_size2 = int(hidden_size / 2)
    hidden_size3 = int(hidden_size / 2)
    return chain(
        concatenate(
            cast(Model[Ragged, Floats2d], reduce_last()),
            cast(Model[Ragged, Floats2d], reduce_first()),
            reduce_mean(),
            reduce_max(),
        ), Maxout(nO=hidden_size, nP=3, normalize=True, dropout=0.0),
        Maxout(nO=hidden_size2, nP=3, normalize=True, dropout=0.0),
        Maxout(nO=hidden_size3, nP=3, normalize=True, dropout=0.0))


@registry.architectures("CustomSpanCategorizer.v2")
def build_spancat_model(
    tok2vec: Model[List[Doc], List[Floats2d]],
    reducer: Model[Ragged, Floats2d],
    scorer: Model[Floats2d, Floats2d],
) -> Model[Tuple[List[Doc], Ragged], Floats2d]:
    """Build a span categorizer model, given a token-to-vector model, a
    reducer model to map the sequence of vectors for each span down to a single
    vector, and a scorer model to map the vectors to probabilities.
    tok2vec (Model[List[Doc], List[Floats2d]]): The tok2vec model.
    reducer (Model[Ragged, Floats2d]): The reducer model.
    scorer (Model[Floats2d, Floats2d]): The scorer model.
    """
    model = chain(
        cast(
            Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
            with_getitem(
                0,
                chain(tok2vec,
                      cast(Model[List[Floats2d], Ragged], list2ragged()))),
        ),
        extract_spans(),
        reducer,
        scorer,
    )
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("reducer", reducer)
    model.set_ref("scorer", scorer)
    return model


# @registry.architectures("spacy.SpanCategorizer.v1")
# def build_spancat_model(
#     tok2vec: Model[List[Doc], List[Floats2d]],
#     reducer: Model[Ragged, Floats2d],
#     scorer: Model[Floats2d, Floats2d],
# ) -> Model[Tuple[List[Doc], Ragged], Floats2d]:
#     """Build a span categorizer model, given a token-to-vector model, a
#     reducer model to map the sequence of vectors for each span down to a single
#     vector, and a scorer model to map the vectors to probabilities.
#     tok2vec (Model[List[Doc], List[Floats2d]]): The tok2vec model.
#     reducer (Model[Ragged, Floats2d]): The reducer model.
#     scorer (Model[Floats2d, Floats2d]): The scorer model.
#     """
#     model = chain(
#         cast(
#             Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
#             with_getitem(
#                 0,
#                 chain(tok2vec,
#                       cast(Model[List[Floats2d], Ragged], list2ragged()))),
#         ),
#         extract_spans(),
#         reducer,
#         scorer,
#     )
#     model.set_ref("tok2vec", tok2vec)
#     model.set_ref("reducer", reducer)
#     model.set_ref("scorer", scorer)
#     return model