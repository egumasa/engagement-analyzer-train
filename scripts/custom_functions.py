from functools import partial
from pathlib import Path
from typing import Iterable, Callable, Optional
import spacy
from spacy.training import Example
from spacy.tokens import DocBin, Doc

from typing import List, Tuple, cast
from thinc.layers.chain import init as init_chain
from thinc.api import Model, with_getitem, chain, list2ragged, Logistic, Softmax, clone, LayerNorm, ParametricAttention, Dropout
from thinc.api import Maxout, Mish, Linear, Gelu, concatenate, glorot_uniform_init, PyTorchLSTM, residual
from thinc.api import reduce_mean, reduce_max, reduce_first, reduce_last, reduce_sum
from thinc.types import Ragged, Floats2d

from spacy.util import registry
from spacy.tokens import Doc
from spacy.ml.extract_spans import extract_spans

## For initializing parametric attention
from spacy.ml.models.tok2vec import get_tok2vec_width
# @registry.layers("spacy.LinearLogistic.v1")
# def build_linear_logistic(nO=None, nI=None) -> Model[Floats2d, Floats2d]:
#     """An output layer for multi-label classification. It uses a linear layer
#     followed by a logistic activation.
#     """
#     return chain(Linear(nO=nO, nI=nI, init_W=glorot_uniform_init), Logistic())


# from typing import Callable, Optional, Tuple, cast

# from thinc.config import registry
# from thinc.model import Model
# from thinc.types import Floats2d, Ragged
# from thinc.util import get_width
# from thinc.layers.noop import noop

# InT = Ragged
# OutT = Ragged

# KEY_TRANSFORM_REF: str = "key_transform"

# # @registry.layers("ParametricAttention.v2")
# def ParametricAttention_v2(

#     key_transform: Optional[Model[Floats2d, Floats2d]] = None,
#     nO: Optional[int] = None
# ) -> Model[InT, OutT]:
#     if key_transform is None:
#         key_transform = noop()

#     """Weight inputs by similarity to a learned vector"""
#     return Model(
#         "para-attn",
#         forward,
#         init=init,
#         params={"Q": None},
#         dims={"nO": nO},
#         refs={KEY_TRANSFORM_REF: key_transform},
#         layers=[key_transform],
#     )


# def forward(model: Model[InT, OutT], Xr: InT, is_train: bool) -> Tuple[OutT, Callable]:
#     Q = model.get_param("Q")
#     key_transform = model.get_ref(KEY_TRANSFORM_REF)

#     attention, bp_attention = _get_attention(
#         model.ops, Q, key_transform, Xr.dataXd, Xr.lengths, is_train
#     )
#     output, bp_output = _apply_attention(model.ops, attention, Xr.dataXd, Xr.lengths)

#     def backprop(dYr: OutT) -> InT:
#         dX, d_attention = bp_output(dYr.dataXd)
#         dQ, dX2 = bp_attention(d_attention)
#         model.inc_grad("Q", dQ.ravel())
#         dX += dX2
#         return Ragged(dX, dYr.lengths)

#     return Ragged(output, Xr.lengths), backprop


# def init(
#     model: Model[InT, OutT], X: Optional[InT] = None, Y: Optional[OutT] = None
# ) -> None:
#     key_transform = model.get_ref(KEY_TRANSFORM_REF)
#     width = get_width(X) if X is not None else None
#     if width:
#         model.set_dim("nO", width)
#         if key_transform.has_dim("nO"):
#             key_transform.set_dim("nO", width)

#     # Randomly initialize the parameter, as though it were an embedding.
#     Q = model.ops.alloc1f(model.get_dim("nO"))
#     Q += model.ops.xp.random.uniform(-0.1, 0.1, Q.shape)
#     model.set_param("Q", Q)

#     X_array = X.dataXd if X is not None else None
#     Y_array = Y.dataXd if Y is not None else None

#     key_transform.initialize(X_array, Y_array)


# def _get_attention(ops, Q, key_transform, X, lengths, is_train):
#     K, K_bp = key_transform(X, is_train=is_train)

#     attention = ops.gemm(K, ops.reshape2f(Q, -1, 1))
#     attention = ops.softmax_sequences(attention, lengths)

#     def get_attention_bwd(d_attention):
#         d_attention = ops.backprop_softmax_sequences(d_attention, attention, lengths)
#         dQ = ops.gemm(K, d_attention, trans1=True)
#         dY = ops.xp.outer(d_attention, Q)
#         dX = K_bp(dY)
#         return dQ, dX

#     return attention, get_attention_bwd


# def _apply_attention(ops, attention, X, lengths):
#     output = X * attention

#     def apply_attention_bwd(d_output):
#         d_attention = (X * d_output).sum(axis=1, keepdims=True)
#         dX = d_output * attention
#         return dX, d_attention

#     return output, apply_attention_bwd



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


# @registry.architectures("Attention_SpanCategorizer.v1")
# def build_spancat_attention_model(
#         tok2vec: Model[List[Doc], List[Floats2d]],
#         reducer: Model[Ragged, Floats2d],
#         scorer: Model[Floats2d, Floats2d],
#         LSTMdepth: int = 2,
#         LSTMdropout: float = 0.0,
#         LSTMhidden: int = 200) -> Model[Tuple[List[Doc], Ragged], Floats2d]:
#     """Build a span categorizer model, given a token-to-vector model, a
#     reducer model to map the sequence of vectors for each span down to a single
#     vector, and a scorer model to map the vectors to probabilities.
#     tok2vec (Model[List[Doc], List[Floats2d]]): The tok2vec model.
#     reducer (Model[Ragged, Floats2d]): The reducer model.
#     scorer (Model[Floats2d, Floats2d]): The scorer model.
#     """
#     with Model.define_operators({">>": chain, "|": concatenate}):
#         width = tok2vec.maybe_get_dim("nO")
#         attention_layer = ParametricAttention(width)
#         maxout_layer = Maxout(nO=width, nI=width)
#         norm_layer = LayerNorm(nI=width)
#         cnn_model = (
#             cast(
#         Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
#         with_getitem(
#             0,
#             chain(
#                 tok2vec,
#                 cast(Model[List[Floats2d], Ragged], list2ragged()))))
#             # >> list2ragged()
#             >> attention_layer
#             >> reduce_sum()
#             >> residual(maxout_layer >> norm_layer >> Dropout(0.0))
#         )

#     embedding = cast(
#         Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
#         with_getitem(
#             0,
#             chain(
#                 tok2vec,
#                 cast(Model[List[Floats2d], Ragged], list2ragged()))))
#     # LSTM = PyTorchLSTM(nO = None, nI= None, bi = True, depth = LSTMdepth, dropout = LSTMdropout)

#     model = chain(
#         concatenate(embedding, cnn_model),
#         extract_spans(),
#         reducer,
#         scorer,
#     )
#     model.set_ref("tok2vec", tok2vec)
#     model.set_ref("reducer", reducer)
#     model.set_ref("scorer", scorer)
#     return model


@registry.architectures("Attention_SpanCategorizer.v1")
def build_spancat_LSTM_model(
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
    width = tok2vec.maybe_get_dim("nO")
    embedding = cast(
        Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
        with_getitem(
            0,
            chain(
                tok2vec,
                ParametricAttention(nO = 768),
                cast(Model[List[Floats2d], Ragged], list2ragged()))))
    

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

@registry.architectures("Attention_SpanCategorizer.v2")
def build_spancat_LSTM_model(
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
    width = tok2vec.maybe_get_dim("nO")
    # embedding = cast(
    #     Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
    #     with_getitem(
    #         0,
    #         chain(
    #             tok2vec,
    #             cast(Model[List[Floats2d], Ragged], list2ragged()))))
    # LSTM = PyTorchLSTM(nO = None, nI= None, bi = True, depth = LSTMdepth, dropout = LSTMdropout)

    model = chain(
        tok2vec,
        list2ragged(),
        ParametricAttention(width),
        extract_spans(),
        reducer,
        scorer,
    )
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("reducer", reducer)
    model.set_ref("scorer", scorer)
    return model

@registry.architectures("Attention_SpanCategorizer.v3")
def build_spancat_attention_model(
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
    width = tok2vec.maybe_get_dim("nO")
    embedding = cast(
        Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
        with_getitem(
            0,
            chain(
                tok2vec,
                cast(Model[List[Floats2d], Ragged], list2ragged()))))
    

    model = chain(
        embedding,
        extract_spans(),
        ParametricAttention(nO = width),
        reducer,
        scorer,
    )
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("reducer", reducer)
    model.set_ref("scorer", scorer)
    return model

@registry.architectures("Attention_SpanCategorizer.v4")
def build_spancat_LSTM_model(
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
    width = tok2vec.maybe_get_dim("nO")
    embedding = cast(
        Model[Tuple[List[Doc], Ragged], Tuple[Ragged, Ragged]],
        with_getitem(
            0,
            chain(
                tok2vec,
                cast(Model[List[Floats2d], Ragged], list2ragged()))))

    attention_layer = chain(
                ParametricAttention(nO = width),
                list2ragged())
  

    model = chain(
        embedding,
        attention_layer,
        extract_spans(),
        reducer,
        scorer,
    )
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("reducer", reducer)
    model.set_ref("scorer", scorer)
    return model


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

@registry.architectures("SpanCatParametricAttention.v1")
def build_textcat_parametric_attention_v1(
    tok2vec: Model[List[Doc], List[Floats2d]],
    exclusive_classes: bool = False,
    nO: Optional[int] = None,
) -> Model[List[Doc], Floats2d]:
    width = tok2vec.maybe_get_dim("nO")
    parametric_attention = _build_parametric_attention_with_residual_nonlinear(
        tok2vec=tok2vec,
        nonlinear_layer=Maxout(nI=width, nO=width),
        key_transform=Gelu(nI=width, nO=width),
    )
    with Model.define_operators({">>": chain}):
        if exclusive_classes:
            output_layer = Softmax(nO=nO)
        else:
            output_layer = Linear(nO=nO, init_W=glorot_uniform_init) >> Logistic()
        model = parametric_attention >> output_layer
    if model.has_dim("nO") is not False and nO is not None:
        model.set_dim("nO", cast(int, nO))
    model.set_ref("output_layer", output_layer)

    return model


def _build_parametric_attention_with_residual_nonlinear(
    *,
    tok2vec: Model[List[Doc], List[Floats2d]],
    nonlinear_layer: Model[Floats2d, Floats2d],
    key_transform: Optional[Model[Floats2d, Floats2d]] = None,
) -> Model[List[Doc], Floats2d]:
    with Model.define_operators({">>": chain, "|": concatenate}):
        width = tok2vec.maybe_get_dim("nO")
        attention_layer = ParametricAttention(nO=width)
        norm_layer = LayerNorm(nI=width)
        parametric_attention = (
            tok2vec
            >> list2ragged()
            >> attention_layer
            >> reduce_sum()
            >> residual(nonlinear_layer >> norm_layer >> Dropout(0.0))
        )

        parametric_attention.init = _init_parametric_attention_with_residual_nonlinear

        parametric_attention.set_ref("tok2vec", tok2vec)
        parametric_attention.set_ref("attention_layer", attention_layer)
        # parametric_attention.set_ref("key_transform", key_transform)
        parametric_attention.set_ref("nonlinear_layer", nonlinear_layer)
        parametric_attention.set_ref("norm_layer", norm_layer)

        return parametric_attention

def _init_parametric_attention_with_residual_nonlinear(model, X, Y) -> Model:
    # When tok2vec is lazily initialized, we need to initialize it before
    # the rest of the chain to ensure that we can get its width.
    tok2vec = model.get_ref("tok2vec")
    tok2vec.initialize(X)

    tok2vec_width = get_tok2vec_width(model)
    model.get_ref("attention_layer").set_dim("nO", tok2vec_width)
    # model.get_ref("key_transform").set_dim("nI", tok2vec_width)
    # model.get_ref("key_transform").set_dim("nO", tok2vec_width)
    model.get_ref("nonlinear_layer").set_dim("nI", tok2vec_width)
    model.get_ref("nonlinear_layer").set_dim("nO", tok2vec_width)
    model.get_ref("norm_layer").set_dim("nI", tok2vec_width)
    model.get_ref("norm_layer").set_dim("nO", tok2vec_width)
    init_chain(model, X, Y)
    return model

@registry.architectures("SpanCatEnsemble.v2")
def build_text_classifier_v2(
    tok2vec: Model[List[Doc], List[Floats2d]],
    nO: Optional[int] = None,
) -> Model[List[Doc], Floats2d]:
    # TODO: build the model with _build_parametric_attention_with_residual_nonlinear
    # in spaCy v4. We don't do this in spaCy v3 to preserve model
    # compatibility.
    
    with Model.define_operators({">>": chain, "|": concatenate}):
        width = tok2vec.maybe_get_dim("nO")
        attention_layer = ParametricAttention(width)
        maxout_layer = Maxout(nO=width, nI=width)
        norm_layer = LayerNorm(nI=width)
        cnn_model = (
            tok2vec
            >> list2ragged()
            >> attention_layer
            >> reduce_sum()
            >> residual(maxout_layer >> norm_layer >> Dropout(0.0))
        )

        nO_double = nO * 2 if nO else None
        if exclusive_classes:
            output_layer = Softmax(nO=nO, nI=nO_double)
        else:
            output_layer = Linear(nO=nO, nI=nO_double) >> Logistic()
        model = cnn_model >> output_layer
        model.set_ref("tok2vec", tok2vec)
    if model.has_dim("nO") is not False and nO is not None:
        model.set_dim("nO", cast(int, nO))

    model.set_ref("attention_layer", attention_layer)
    model.set_ref("maxout_layer", maxout_layer)
    model.set_ref("norm_layer", norm_layer)


    model.init = init_ensemble_textcat  # type: ignore[assignment]
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


@registry.layers("sum_reducer.v1")
def build_sum_reducer(hidden_size: int,
                            dropout: float = 0.0,
                            depth: int = 1) -> Model[Ragged, Floats2d]:
    """Reduce sequences by concatenating their mean and max pooled vectors,
    and then combine the concatenated vectors with a hidden layer.
    """
    return chain(
        reduce_sum(),
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


@registry.layers("attention_reducer.v1")
def build_mean_max_reducer1(hidden_size: int,
                            dropout: float = 0.0,
                            depth: int = 1) -> Model[Ragged, Floats2d]:
    """
    """

    with Model.define_operators({">>": chain, "|": concatenate}):
        width = tok2vec.maybe_get_dim("nO")
        attention_layer = ParametricAttention(width)
        maxout_layer = Maxout(nO=width, nI=width)
        norm_layer = LayerNorm(nI=width)
        cnn_model = (
            tok2vec
            >> list2ragged()
            >> attention_layer
            >> reduce_sum()
            >> residual(maxout_layer >> norm_layer >> Dropout(0.0))
        )


    return chain(
        concatenate(
            cast(Model[Ragged, Floats2d], reduce_last()),
            cast(Model[Ragged, Floats2d], reduce_first()),
            reduce_mean(),
            reduce_max(),
        ),
        clone(Maxout(nO=hidden_size, normalize=True, dropout=dropout), depth),
    )


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



