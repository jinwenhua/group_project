import torch
import torch.nn as nn
import torch.nn.functional as F


######################################################################
# Define Encoder
# --------------
#
# We implement our encoder’s RNN with the ``torch.nn.GRU`` module which we
# feed a batch of sentences (vectors of word embeddings) and it internally
# iterates through the sentences one token at a time calculating the
# hidden states. We initialize this module to be bidirectional, meaning
# that we have two independent GRUs: one that iterates through the
# sequences in chronological order, and another that iterates in reverse
# order. We ultimately return the sum of these two GRUs’ outputs. Since
# our model was trained using batching, our ``EncoderRNN`` model’s
# ``forward`` function expects a padded input batch. To batch
# variable-length sentences, we allow a maximum of *MAX_LENGTH* tokens in
# a sentence, and all sentences in the batch that have less than
# *MAX_LENGTH* tokens are padded at the end with our dedicated *PAD_token*
# tokens. To use padded batches with a PyTorch RNN module, we must wrap
# the forward pass call with ``torch.nn.utils.rnn.pack_padded_sequence``
# and ``torch.nn.utils.rnn.pad_packed_sequence`` data transformations.
# Note that the ``forward`` function also takes an ``input_lengths`` list,
# which contains the length of each sentence in the batch. This input is
# used by the ``torch.nn.utils.rnn.pack_padded_sequence`` function when
# padding.
#
# TorchScript Notes:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Since the encoder’s ``forward`` function does not contain any
# data-dependent control flow, we will use **tracing** to convert it to
# script mode. When tracing a module, we can leave the module definition
# as-is. We will initialize all models towards the end of this document
# before we run evaluations.
#

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden

    ######################################################################
# Define Decoder’s Attention Module
# ---------------------------------
#
# Next, we’ll define our attention module (``Attn``). Note that this
# module will be used as a submodule in our decoder model. Luong et
# al. consider various “score functions”, which take the current decoder
# RNN output and the entire encoder output, and return attention
# “energies”. This attention energies tensor is the same size as the
# encoder output, and the two are ultimately multiplied, resulting in a
# weighted tensor whose largest values represent the most important parts
# of the query sentence at a particular time-step of decoding.
#

# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
    
    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


######################################################################
# Define Decoder
# --------------
#
# Similarly to the ``EncoderRNN``, we use the ``torch.nn.GRU`` module for
# our decoder’s RNN. This time, however, we use a unidirectional GRU. It
# is important to note that unlike the encoder, we will feed the decoder
# RNN one word at a time. We start by getting the embedding of the current
# word and applying a
# `dropout <https://pytorch.org/docs/stable/nn.html?highlight=dropout#torch.nn.Dropout>`__.
# Next, we forward the embedding and the last hidden state to the GRU and
# obtain a current GRU output and hidden state. We then use our ``Attn``
# module as a layer to obtain the attention weights, which we multiply by
# the encoder’s output to obtain our attended encoder output. We use this
# attended encoder output as our ``context`` tensor, which represents a
# weighted sum indicating what parts of the encoder’s output to pay
# attention to. From here, we use a linear layer and softmax normalization
# to select the next word in the output sequence.

# TorchScript Notes:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Similarly to the ``EncoderRNN``, this module does not contain any
# data-dependent control flow. Therefore, we can once again use
# **tracing** to convert this model to TorchScript after it
# is initialized and its parameters are loaded.
#

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
