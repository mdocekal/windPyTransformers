# -*- coding: UTF-8 -*-
""""
Created on 19.03.20
Module that uses convolution and attention.

:author:     Martin Dočekal
"""

import torch
from typing import List, Optional

from transformers import PreTrainedModel, PretrainedConfig

from windpytorchutils.nn.attention_pooler import AttentionPooler


class Layer(torch.nn.Module):
    """
    Layer that selects convolution filter/s according to input.

    Visualization:
        ---------                         -------------
    --- | TOKEN | ---------- tokenSize--> | ATTENTION |
    |   ---------                         -------------
    |     +-- INP_LEN x inputOutputSize --|     |
    |     |                                     |
    |     |                               inputOutputSize
    |     |                                     |
    |     |                            --------------------
    |     |                            | LINEAR TRANSFORM |
    |     |                            |     (decide)     |
    |     |                            --------------------
    |     |                            |      SOFTMAX     |
    |     |                            --------------------
    |     |                                     |
    |     |   each token         weights for filter sizes (size: len(kernels)) ----------|
    |     |      |                                                                       |
    |   -----    |        ----------------------------                                ------------------
    |   | I | -------->   |             |            |                                |                |
    |   | N | -------->   | convolution |            |-KERNELS x (CHANNELSxSEQUENCE)->|     matrix     |
    |   | P | -------->   |   filters   | activation |                                | multiplication |
    |   | U | -------->   |             |            |                                |                |
    |   | T | -------->   ----------------------------                                ------------------
    |                                                                                          |
    |                                                                       SEQUENCE X CHANNELS (Weighted among kernels)
    |                                                                                   = SEQUENCE X HIDDEN
    |                                                                                          |
    |                                                                --------------------------------------------------
    |------------------------------------------------------------->  | concatenate token to each sequence hidden state|
                                                                     --------------------------------------------------
                                                                                               |
                                                                                  SEQUENCE X (tokenSize+HIDDEN)
                                                                                               |
                                                                                      --------------------
                                                                                      | LINEAR TRANSFORM |
                                                                                      --------------------
                                                                                      |   normalization  |
                                                                                      --------------------
                                                                                               |
                                                                                OUTPUT: SEQUENCE X layer_output_size
    """

    def __init__(self, tokenSize: int, inputOutputSize: int, hiddenSize: int, kernels: Optional[List[int]] = None,
                 outputSize: Optional[int] = None, layerNormEps: float = 1e-12):
        """
        Layer initialization.

        :param tokenSize: Number of dimensions of a token.
        :type tokenSize: int
        :param inputOutputSize: Size of input/output embeddings.
        :type inputOutputSize: int
        :param hiddenSize: Size of internal embeddings.
        :type hiddenSize: int
        :param kernels: Defines bank of kernels.
        :type kernels: Optional[List[int]]
        :param outputSize: Size of output embeddings.
            Usually the input output size are the same.
        :type outputSize: Optional[int]
        :param layerNormEps: Epsilon for normalization layer.
        :type layerNormEps: float
        """

        super().__init__()

        if kernels is None:
            kernels = [3, 5, 7, 17, 31]

        if outputSize is None:
            outputSize = inputOutputSize

        self.attention = AttentionPooler(inputOutputSize, hiddenStateSize=tokenSize)
        self.decide = torch.nn.Linear(inputOutputSize, len(kernels))  # we want a score for each channel
        self.softmax = torch.nn.Softmax(dim=1)

        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(inputOutputSize, hiddenSize, kernelSize,
                                                          padding=int((kernelSize - 1) / 2)) for kernelSize in kernels])
        self.convActivation = torch.nn.LeakyReLU()

        self.contextDepTrans = torch.nn.Linear(tokenSize + hiddenSize, outputSize)
        self.norm = torch.nn.LayerNorm(outputSize, eps=layerNormEps)

    def forward(self, token: torch.Tensor, inputElements: torch.Tensor, attentionMask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass trough layer.

        :param token: Token that is used as the hidden state for the attention.
            Size: BATCH X tokenSize
        :type token: torch.Tensor
        :param inputElements: The input for that layer.
            Size: BATCH X SEQUENCE X inputSize
        :type inputElements: torch.Tensor
        :param attentionMask: Mask for that parts you want to hide from attention.
            Size: BATCH X SEQUENCE
        :type attentionMask: torch.Tensor
        :return: Output of that layer.
            SIZE: BATCH X SEQUENCE X outputSize
        :rtype: torch.Tensor
        """

        attentionPooled = self.attention(token, inputElements, attentionMask=attentionMask)  # BATCH x POOLED TENSOR
        softDecisions = self.softmax(self.decide(attentionPooled))  # BATCH x PROBABILITIES OF CHANNELS

        inputElementsPermutedForConvolution = inputElements.permute(0, 2, 1)  # BATCH x EMBEDDING DIM x SEQUENCE

        channels = []  # THERE WILL BE A LIST OF: BATCH x 1 x CHANNELS x SEQUENCE
        for i, conv in enumerate(self.convs):
            channels.append(self.convActivation(conv(inputElementsPermutedForConvolution)).unsqueeze(1))

        channels = torch.cat(channels, dim=1)  # BATCH x KERNELS x CHANNELS x SEQUENCE

        # apply soft decision (predicted weights of the kernels) on the kernels

        flattenedChannels = channels.view(channels.shape[0], channels.shape[1],
                                          -1)  # BATCH x KERNELS x (CHANNELSxSEQUENCE) just to be able to use torch.bmm

        hidden = torch.bmm(softDecisions.unsqueeze(1),
                           flattenedChannels)  # BATCH x 1 x (CHANNELS(Weighted among kernels)xSEQUENCE) [1 - means reasult of weighted average among the kernels]

        hidden = hidden.view(channels.shape[0], channels.shape[2], -1).permute(0, 2,
                                                                               1)  # BATCH x SEQUENCE X CHANNELS (Weighted among kernels)

        # concatenate token to each sequence feature element
        tokenPlusHiddenEmbedding = torch.cat((token.unsqueeze(dim=1).expand(token.shape[0], hidden.shape[1],
                                                                            token.shape[1]), hidden),
                                             dim=2)  # BATCH x concatenated token with hidden

        # apply the linear transformation and normalization
        transformed = self.contextDepTrans(tokenPlusHiddenEmbedding)
        transformed = self.norm(transformed)

        return transformed


class ConvolverConfig(PretrainedConfig):
    """
    Configuration for `Convolver`.
    """

    def __init__(self, vocab_size: int = 30000, embedding_size: int = 128, token_size: int = 128,
                 hidden_size: int = 2,
                 layer_output_size: int = 768, kernels: Optional[List[int]] = None, num_hidden_layers: int = 2,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-5, **kwargs):
        """

        :param vocab_size: Number of elements in vocabulary.
        :type vocab_size: int
        :param token_size: Number of dimensions of input token.
        :type token_size: int
        :param embedding_size: Number of dimensions of an embedding
        :type embedding_size: int
        :param hidden_size: Number of convolution kernels of same size, which determines the hidden size of model.
        :type hidden_size: int
        :param layer_output_size: Determines number of dimensions for each output token vector.
        :type layer_output_size: int
        :param kernels: Sizes of convolution kernels.
        :type kernels: Optional[List[int]]
        :param num_hidden_layers: Number of layers
        :type num_hidden_layers: int
        :param initializer_range: STDEV for the initialization of weights.
        :type initializer_range: float
        :param layer_norm_eps: Epsilon for normalization layer.
        :type layer_norm_eps: float
        :param kwargs: This other arguments will be passed to the base class.
        :type kwargs:
        """
        super(ConvolverConfig, self).__init__(**kwargs)

        if kernels is None:
            kernels = [3, 5, 7, 17, 31]

        self.vocab_size = vocab_size
        self.token_size = token_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layer_output_size = layer_output_size
        self.kernels = kernels
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class ConvolverPreTrained(PreTrainedModel):
    """
    An abstract class to handle weights initialization.
    """

    def _init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Convolver(ConvolverPreTrained):
    """
    Module that uses convolution and attention.

    Visualization for 3 layers:

                   SEQUENCE x hiddenSize
                     |  |  |  |  |  |
        ------------------------------------------
        |                LAYER                   |
        ------------------------------------------
        |                        |  |  |  |  |  |
        |                        |  |  |  |  |  |
   token (size: hiddenSize)      |  |  |  |  |  |
        |                        |  |  |  |  |  |
    ----------------------       |  |  |  |  |  |
    | MAX POOL over time | <-- SEQUENCE x hiddenSize
    ----------------------       |  |  |  |  |  |
        ------------------------------------------
        |                LAYER                   |
        ------------------------------------------
        |                        |  |  |  |  |  |
        |                        |  |  |  |  |  |
   token (size: hiddenSize)      |  |  |  |  |  |
        |                        |  |  |  |  |  |
    ----------------------       |  |  |  |  |  |
    | MAX POOL over time | <-- SEQUENCE x hiddenSize
    ----------------------       |  |  |  |  |  |
        ------------------------------------------
        |                LAYER                   |
        ------------------------------------------
        |                        |  |  |  |  |  |
   token tokenSize           SEQUENCE x embeddingSize
    """

    def __init__(self, config: ConvolverConfig):
        """
        Initialization of a model.

        :param config: configuration
        :type config: ConvolverConfig
        """

        super(Convolver, self).__init__(config)

        self.config = config

        self.embeddings = torch.nn.Embedding(config.vocab_size, config.embedding_size)

        self.layers = torch.nn.ModuleList([Layer(tokenSize=config.token_size, inputOutputSize=config.embedding_size,
                                                 hiddenSize=config.hidden_size, kernels=config.kernels,
                                                 outputSize=config.layer_output_size,
                                                 layerNormEps=config.layer_norm_eps)]
                                          + [Layer(tokenSize=config.layer_output_size,
                                                   inputOutputSize=config.layer_output_size,
                                                   hiddenSize=config.hidden_size, kernels=config.kernels,
                                                   layerNormEps=config.layer_norm_eps)
                                             for _ in range(config.num_hidden_layers - 1)])

        self.init_weights()

    def forward(self, inputSequence: torch.Tensor, token: Optional[torch.Tensor] = None,
                attentionMask: Optional[torch.Tensor] = None) \
            -> torch.Tensor:
        """
        Performs forward phase on the model.

        :param inputSequence: Parsed input tokens.
        :type inputSequence: torch.Tensor
        :param token: Token that is used in the first layer attention.
            If you will not pass this token than this token will be the same as first inputSequence token.
        :type token: Optional[torch.Tensor]
        :param attentionMask: Mask to avoid attention on (padding) token indices. 1 NOT  MASKED, 0 for MASKED.
        :type attentionMask: torch.Tensor
        :return: The output of convolver in form of sequence:
            BATCH x TIME x HIDDEN_SIZE
        :rtype: torch.Tensor
        """

        wordEmbeddings = self.embeddings(inputSequence)

        if token is None:
            token = self.embeddings(inputSequence[:, 0])

        transformed = self.layers[0](token, wordEmbeddings, attentionMask)
        for layer in self.layers[1:]:
            newToken = torch.max_pool1d(transformed.permute(0, 2, 1), transformed.shape[1]).squeeze(
                2)  # we take the max over time dimension for each feature
            transformed = layer(newToken, transformed, attentionMask)

        return transformed
