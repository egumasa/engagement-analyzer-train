from functools import partial
from pathlib import Path
from typing import Iterable, Callable
import spacy
from spacy.training import Example
from spacy.tokens import DocBin, Doc

from typing import List, Tuple, cast
from thinc.api import Model, with_getitem, chain, list2ragged, Logistic, clone, LayerNorm
from thinc.api import Maxout, Mish, Linear, Gelu, concatenate, glorot_uniform_init, PyTorchLSTM, residual
from thinc.api import reduce_mean, reduce_max, reduce_first, reduce_last, reduce_sum
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

@registry.architectures("CustomSpanCategorizer.v2")
def build_spancat_model(
    tok2vec: Model[List[Doc], List[Floats2d]],
    reducer1: Model[Ragged, Floats2d],
    reducer2: Model[Ragged, Floats2d],
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
        concatenate(reducer1, reducer2),
        scorer,
    )
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("reducer1", reducer1)
    model.set_ref("reducer2", reducer2)
    model.set_ref("scorer", scorer)
    return model


@registry.architectures("LSTM_SpanCategorizer.v1")
def build_spancat_LSTM_model(
        tok2vec: Model[List[Doc], List[Floats2d]],
        reducer: Model[Ragged, Floats2d],
        scorer: Model[Floats2d, Floats2d],
        LSTMdepth: int = 2,
        LSTMdropout: float = 0.0,
        LSTMhidden: int = 200) -> Model[Tuple[List[Doc], Ragged], Floats2d]:
    """Build a span categorizer model, given a token-to-vector model, a
    reducer model to map the sequence of vectors for each span down to a single
    vector, and a scorer model to map the vectors to probabilities.
    tok2vec (Model[List[Doc], List[Floats2d]]): The tok2vec model.
    reducer (Model[Ragged, Floats2d]): The reducer model.
    scorer (Model[Floats2d, Floats2d]): The scorer model.
    """
    embedding = cast(
        Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
        with_getitem(
            0,
            chain(
                tok2vec,
                PyTorchLSTM(nI=768,
                            nO=LSTMhidden,
                            bi=True,
                            depth=LSTMdepth,
                            dropout=LSTMdropout),
                cast(Model[List[Floats2d], Ragged], list2ragged()))))
    # LSTM = PyTorchLSTM(nO = None, nI= None, bi = True, depth = LSTMdepth, dropout = LSTMdropout)

    model = chain(
        embedding,
        extract_spans(),
        reducer,
        scorer,
    )
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("reducer", reducer)
    model.set_ref("scorer", scorer)
    return model

@registry.architectures("Ensemble_SpanCategorizer.v1")
def build_spancat_model3(
    tok2vec: Model[List[Doc], List[Floats2d]],
    tok2vec_trf: Model[List[Doc], List[Floats2d]],
    reducer1: Model[Ragged, Floats2d],
    reducer2: Model[Ragged, Floats2d],
    scorer: Model[Floats2d, Floats2d],
) -> Model[Tuple[List[Doc], Ragged], Floats2d]:
    """Build a span categorizer model, given a token-to-vector model, a
    reducer model to map the sequence of vectors for each span down to a single
    vector, and a scorer model to map the vectors to probabilities.
    tok2vec (Model[List[Doc], List[Floats2d]]): The tok2vec model.
    reducer (Model[Ragged, Floats2d]): The reducer model.
    scorer (Model[Floats2d, Floats2d]): The scorer model.
    """
    trainable_trf = cast(
        Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
        with_getitem(
            0,
            chain(tok2vec, cast(Model[List[Floats2d], Ragged],
                                list2ragged()))),
    )
    en_core_web_trf = cast(
        Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
        with_getitem(
            0,
            chain(tok2vec_trf,
                  cast(Model[List[Floats2d], Ragged], list2ragged()))),
    )
    reduce_trainable = chain(trainable_trf, extract_spans(), reducer1)
    reduce_default = chain(en_core_web_trf, extract_spans(), reducer2)
    model = chain(
        concatenate(reduce_trainable, reduce_default),
        # Mish(),
        # LayerNorm(),
        scorer,
    )
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("tok2vec_trf", tok2vec_trf)
    model.set_ref("reducer1", reducer1)
    model.set_ref("reducer2", reducer2)
    model.set_ref("scorer", scorer)
    return model


@registry.architectures("Ensemble_SpanCategorizer.v2")
def build_spancat_model3(
    tok2vec: Model[List[Doc], List[Floats2d]],
    tok2vec_trf: Model[List[Doc], List[Floats2d]],
    reducer1: Model[Ragged, Floats2d],
    reducer2: Model[Ragged, Floats2d],
    scorer: Model[Floats2d, Floats2d],
    LSTMhidden: int = 200,
    LSTMdepth: int = 1,
    LSTMdropout: float = 0.0,
) -> Model[Tuple[List[Doc], Ragged], Floats2d]:
    """Build a span categorizer model, given a token-to-vector model, a
    reducer model to map the sequence of vectors for each span down to a single
    vector, and a scorer model to map the vectors to probabilities.
    tok2vec (Model[List[Doc], List[Floats2d]]): The tok2vec model.
    reducer (Model[Ragged, Floats2d]): The reducer model.
    scorer (Model[Floats2d, Floats2d]): The scorer model.
    """
    trainable_trf = cast(
        Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
        with_getitem(
            0,
            chain(tok2vec, cast(Model[List[Floats2d], Ragged],
                                list2ragged()))),
    )
    en_core_web_trf = cast(
        Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
        with_getitem(
            0,
            chain(
                tok2vec_trf,
                PyTorchLSTM(nI=768,
                            nO=LSTMhidden,
                            bi=True,
                            depth=LSTMdepth,
                            dropout=LSTMdropout),
                cast(Model[List[Floats2d], Ragged], list2ragged()))),
    )
    reduce_trainable = chain(trainable_trf, extract_spans(), reducer1)
    reduce_default = chain(en_core_web_trf, extract_spans(), reducer2)
    model = chain(
        concatenate(reduce_trainable, reduce_default),
        # Mish(),
        # LayerNorm(),
        scorer,
    )
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("tok2vec_trf", tok2vec_trf)
    model.set_ref("reducer1", reducer1)
    model.set_ref("reducer2", reducer2)
    model.set_ref("scorer", scorer)
    return model


@registry.architectures("Ensemble_SpanCategorizer.v4")
def build_spancat_model3(
    tok2vec: Model[List[Doc], List[Floats2d]],
    tok2vec_trf: Model[List[Doc], List[Floats2d]],
    reducer1: Model[Ragged, Floats2d],
    reducer2: Model[Ragged, Floats2d],
    scorer: Model[Floats2d, Floats2d],
    LSTMhidden: int = 200,
    LSTMdepth: int = 1,
    LSTMdropout: float = 0.0,
) -> Model[Tuple[List[Doc], Ragged], Floats2d]:
    """Build a span categorizer model, given a token-to-vector model, a
    reducer model to map the sequence of vectors for each span down to a single
    vector, and a scorer model to map the vectors to probabilities.
    tok2vec (Model[List[Doc], List[Floats2d]]): The tok2vec model.
    reducer (Model[Ragged, Floats2d]): The reducer model.
    scorer (Model[Floats2d, Floats2d]): The scorer model.
    """
    trainable_trf = cast(
        Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
        with_getitem(
            0,
            chain(tok2vec, cast(Model[List[Floats2d], Ragged],
                                list2ragged()))),
    )
    en_core_web_trf = cast(
        Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
        with_getitem(
            0,
            chain(
                tok2vec_trf,
                PyTorchLSTM(nI=768,
                            nO=LSTMhidden,
                            bi=True,
                            depth=LSTMdepth,
                            dropout=LSTMdropout),
                cast(Model[List[Floats2d], Ragged], list2ragged()))),
    )
    reduce_trainable = chain(trainable_trf, extract_spans(), reducer1)
    reduce_default = chain(en_core_web_trf, extract_spans(), reducer2)
    model = chain(
        concatenate(reduce_trainable, reduce_default),
        Mish(nO = 128),
        LayerNorm(),
        scorer,
    )
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("tok2vec_trf", tok2vec_trf)
    model.set_ref("reducer1", reducer1)
    model.set_ref("reducer2", reducer2)
    model.set_ref("scorer", scorer)
    return model


@registry.layers("mean_max_reducer.v1.5")
def build_mean_max_reducer1(hidden_size: int,
                            dropout: float = 0.0,
                            depth: int = 1) -> Model[Ragged, Floats2d]:
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
        clone(Maxout(nO=hidden_size, normalize=True, dropout=dropout), depth),
    )


# @registry.layers("mean_max_reducer.v2")
# def build_mean_max_reducer2(hidden_size: int,
#                             dropout: float = 0.0) -> Model[Ragged, Floats2d]:
#     """Reduce sequences by concatenating their mean and max pooled vectors,
#     and then combine the concatenated vectors with a hidden layer.
#     """
#     return chain(
#         concatenate(
#             cast(Model[Ragged, Floats2d], reduce_last()),
#             cast(Model[Ragged, Floats2d], reduce_first()),
#             reduce_mean(),
#             reduce_mean(),
#             reduce_max(),
#         ),
#         Maxout(nO=hidden_size, normalize=True, dropout=dropout),
#         )


@registry.layers("Gelu_mean_max_reducer.v1")
def build_mean_max_reducer_gelu(hidden_size: int,
                                dropout: float = 0.0,
                                depth: int = 1) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    gelu_unit = Gelu(nO=hidden_size, normalize=True, dropout=dropout)
    return chain(
        concatenate(
            cast(Model[Ragged, Floats2d], reduce_last()),
            cast(Model[Ragged, Floats2d], reduce_first()),
            reduce_mean(),
            reduce_max(),
        ),
        clone(gelu_unit, depth),
    )


@registry.layers("Mish_mean_max_reducer.v1")
def build_mean_max_reducer3(hidden_size: int,
                            dropout: float = 0.0,
                            depth: int = 4) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    mish_unit = Mish(nO=hidden_size, normalize=True, dropout=dropout)
    return chain(
        concatenate(
            cast(Model[Ragged, Floats2d], reduce_last()),
            cast(Model[Ragged, Floats2d], reduce_first()),
            reduce_mean(),
            reduce_max(),
        ),
        clone(mish_unit, depth),
    )

@registry.layers("Maxout_mean_max_reducer.v2")
def build_mean_max_reducer3(hidden_size: int,
                            dropout: float = 0.0,
                            depth: int = 4) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    maxout_unit = Maxout(nO=hidden_size, normalize=True, dropout=dropout)
    return chain(
        concatenate(
            cast(Model[Ragged, Floats2d], reduce_last()),
            cast(Model[Ragged, Floats2d], reduce_first()),
            reduce_mean(),
            reduce_max(),
        ),
        clone(maxout_unit, depth),
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
            reduce_mean(),
            reduce_max(),
        ), 
        Maxout(nO=hidden_size, normalize=True, dropout=dropout),
        )

@registry.layers("two_way_reducer.v1")
def build_two_way_reducer(hidden_size: int,
                          dropout: float = 0.0) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    default_reducer = concatenate(
        cast(Model[Ragged, Floats2d], reduce_last()),
        cast(Model[Ragged, Floats2d], reduce_first()),
        reduce_mean(),
        reduce_max(),
    )
    mean_sum_reducer = concatenate(reduce_mean(), reduce_sum())

    return concatenate(
        chain(default_reducer,
              Maxout(nO=hidden_size, normalize=True, dropout=dropout)),
        chain(mean_sum_reducer,
              Maxout(nO=hidden_size // 2, normalize=True, dropout=dropout)))


@registry.layers("Mish_two_way_reducer.v1")
def build_Mish_two_way_reducer(hidden_size: int,
                               dropout: float = 0.0,
                               depth: int = 1) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    default_reducer = concatenate(
        cast(Model[Ragged, Floats2d], reduce_last()),
        cast(Model[Ragged, Floats2d], reduce_first()),
        reduce_mean(),
        reduce_max(),
    )
    mean_sum_reducer = concatenate(reduce_mean(), reduce_sum())

    return concatenate(
        chain(
            default_reducer,
            clone(Mish(nO=hidden_size // 2, normalize=True, dropout=dropout),
                  depth)),
        chain(
            mean_sum_reducer,
            clone(Mish(nO=hidden_size // 2, normalize=True, dropout=dropout),
                  depth)))

@registry.layers("Mish_two_way_reducer.v2")
def build_Mish_two_way_reducer2(hidden_size: int,
                               dropout: float = 0.0,
                               depth: int = 1) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    default_reducer = concatenate(
        cast(Model[Ragged, Floats2d], reduce_last()),
        cast(Model[Ragged, Floats2d], reduce_first()),
        reduce_mean(),
        reduce_max(),
    )
    mean_sum_reducer = concatenate(
        cast(Model[Ragged, Floats2d], reduce_last()),
        cast(Model[Ragged, Floats2d], reduce_first()),
        reduce_mean(),
        reduce_sum(),
    )

    return concatenate(
        chain(
            default_reducer,
            clone(Mish(nO=hidden_size // 2, normalize=True, dropout=dropout),
                  depth)),
        chain(
            mean_sum_reducer,
            clone(Mish(nO=hidden_size // 2, normalize=True, dropout=dropout),
                  depth)))




@registry.layers("three_way_reducer.v3")
def build_mean_max_reducer2(hidden_size: int,
                            dropout: float = 0.0,
                            depth: int = 2) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    default_reducer = concatenate(
                cast(Model[Ragged, Floats2d], reduce_last()),
                cast(Model[Ragged, Floats2d], reduce_first()),
                reduce_mean(),
                reduce_max(),
            )
    mean_sum_reducer = concatenate(
            reduce_mean(),
            reduce_sum())

    return concatenate(chain(default_reducer,
                        Maxout(nO=hidden_size, normalize=True, dropout=dropout)),
                       chain(mean_sum_reducer,
                        Maxout(nO=hidden_size//2, normalize=True, dropout=dropout)),
                       chain(mean_sum_reducer,
                        clone(Maxout(nO=hidden_size//2, normalize=True, dropout=dropout),depth))
                    )

@registry.layers("Maxout_three_way_reducer.v1")
def build_Maxout_three_way_reducer(hidden_size: int,
                                   dropout: float = 0.0,
                                   depth: int = 2) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    default_reducer = concatenate(
        cast(Model[Ragged, Floats2d], reduce_last()),
        cast(Model[Ragged, Floats2d], reduce_first()),
        reduce_mean(),
        reduce_max(),
    )
    mean_sum_reducer = concatenate(reduce_mean(), reduce_sum())

    return concatenate(
        chain(
            default_reducer,
            clone(Maxout(nO=hidden_size // 2, normalize=True, dropout=dropout),
                  depth)),
        chain(mean_sum_reducer,
              Maxout(nO=hidden_size // 4, normalize=True, dropout=dropout)),
        chain(
            mean_sum_reducer,
            clone(Maxout(nO=hidden_size // 4, normalize=True, dropout=dropout),
                  depth)))


@registry.layers("Mish_three_way_reducer.v1")
def build_Mish_three_way_reducer(hidden_size: int,
                                 dropout: float = 0.0,
                                 depth: int = 2) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    default_reducer = concatenate(
        cast(Model[Ragged, Floats2d], reduce_last()),
        cast(Model[Ragged, Floats2d], reduce_first()),
        reduce_mean(),
        reduce_max(),
    )
    mean_sum_reducer = concatenate(reduce_mean(), reduce_sum())

    return concatenate(
        chain(
            default_reducer,
            clone(Mish(nO=hidden_size // 2, normalize=True, dropout=dropout),
                  depth)),
        chain(mean_sum_reducer,
              Mish(nO=hidden_size // 4, normalize=True, dropout=dropout)),
        chain(
            mean_sum_reducer,
            clone(Mish(nO=hidden_size // 4, normalize=True, dropout=dropout),
                  depth)))


@registry.layers("mean_max_reducer.v4")
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


# @registry.architectures("spacy.MaxoutWindowEncoder.v2")
# def MaxoutWindowEncoder(
#     width: int, window_size: int, maxout_pieces: int, depth: int
# ) -> Model[List[Floats2d], List[Floats2d]]:
#     """Encode context using convolutions with maxout activation, layer
#     normalization and residual connections.
#     width (int): The input and output width. These are required to be the same,
#         to allow residual connections. This value will be determined by the
#         width of the inputs. Recommended values are between 64 and 300.
#     window_size (int): The number of words to concatenate around each token
#         to construct the convolution. Recommended value is 1.
#     maxout_pieces (int): The number of maxout pieces to use. Recommended
#         values are 2 or 3.
#     depth (int): The number of convolutional layers. Recommended value is 4.
#     """
#     cnn = chain(
#         expand_window(window_size=window_size),
#         Maxout(
#             nO=width,
#             nI=width * ((window_size * 2) + 1),
#             nP=maxout_pieces,
#             dropout=0.0,
#             normalize=True,
#         ),
#     )
#     model = clone(residual(cnn), depth)
#     model.set_dim("nO", width)
#     receptive_field = window_size * depth
#     return with_array(model, pad=receptive_field)


# @registry.architectures("spacy.MishWindowEncoder.v2")
# def MishWindowEncoder(
#     width: int, window_size: int, depth: int
# ) -> Model[List[Floats2d], List[Floats2d]]:
#     """Encode context using convolutions with mish activation, layer
#     normalization and residual connections.
#     width (int): The input and output width. These are required to be the same,
#         to allow residual connections. This value will be determined by the
#         width of the inputs. Recommended values are between 64 and 300.
#     window_size (int): The number of words to concatenate around each token
#         to construct the convolution. Recommended value is 1.
#     depth (int): The number of convolutional layers. Recommended value is 4.
#     """
#     cnn = chain(
#         expand_window(window_size=window_size),
#         Mish(nO=width, nI=width * ((window_size * 2) + 1), dropout=0.0, normalize=True),
#     )
#     model = clone(residual(cnn), depth)
#     model.set_dim("nO", width)
#     return with_array(model)



