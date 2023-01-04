from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from pre_process import *

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import random
import os


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math



class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size)

        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Output:
        #       hy: of shape (batch_size, hidden_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)


        x_reset, x_upd, x_new = x_t.chunk(3, 0)
        h_reset, h_upd, h_new = h_t.chunk(3, 0)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy




class BidirRecurrentModel(nn.Module):
    def __init__(self, mode, input_size, hidden_size, num_layers, num_words = 0, embedding = None, bias = True):
        super(BidirRecurrentModel, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.num_words = num_words
        self.run_list = []
        self.embedding = embedding

        self.rnn_cell_list = []
        print(math.floor(num_words / 10000) + 1, num_words)
        for _ in range(math.floor(num_words / 10000) + 1):
            if mode == 'GRU':
                self.rnn_cell_list.append(nn.GRUCell(self.input_size,
                                                self.hidden_size,
                                                self.bias))
                for l in range(1, self.num_layers):
                    self.rnn_cell_list.append(nn.GRUCell(self.hidden_size,
                                                    self.hidden_size,
                                                    self.bias))
            else:
                raise ValueError("Invalid RNN mode selected.")

            #self.fc = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, input, hx=None, merge_mode = 'ave'):

        # Input of shape (batch_size, sequence length, input_size)
        #
        # Output of shape (batch_size, output_size)
        max_batch_size = input.size(1)
        sort_num = input.shape[0]
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers * 2, max_batch_size, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers * 2, max_batch_size, self.hidden_size))


        outs = []
        outs_rev = []
        self.run_list = []
        hidden_forward = list()
        for layer in range(self.num_layers):
            hidden_forward.append(h0[layer, :, :])

        hidden_backward = list()
        for layer in range(self.num_layers):
            hidden_backward.append(h0[self.num_layers + layer, :, :])

        if torch.cuda.is_available():
            h_forward_l = Variable(torch.zeros(max_batch_size, self.hidden_size).cuda())
            h_back_l = Variable(torch.zeros(max_batch_size, self.hidden_size).cuda())
        else:
            h_forward_l = Variable(torch.zeros( max_batch_size, self.hidden_size))
            h_back_l = Variable(torch.zeros(max_batch_size, self.hidden_size))
    
        for t in range(sort_num):
            for layer in range(self.num_layers):
                for i in range(max_batch_size):
                    cell_id = math.floor(input[t][i]/ 10000) * self.num_layers 
                    #print(cell_id)
                    if input[t][i] > 0:
                        self.run_list.append(cell_id + layer)
                        if layer == 0:
                            forward_input = self.embedding(input[t][i])
                            backward_input = self.embedding(input[-(t + 1)][i])
                            # Forward net
                            h_forward_l[i] = self.rnn_cell_list[cell_id](forward_input, hidden_forward[layer][i])
                            # Backward net
                            h_back_l[i] = self.rnn_cell_list[cell_id](backward_input, hidden_backward[layer][i])
                        else:
                            # Forward net
                            h_forward_l[i] = self.rnn_cell_list[cell_id + layer](hidden_forward[layer - 1][i], hidden_forward[layer][i])
                            # Backward net
                            h_back_l[i] = self.rnn_cell_list[cell_id + layer](hidden_backward[layer - 1][i], hidden_backward[layer][i])

                hidden_forward[layer] = h_forward_l
                hidden_backward[layer] = h_back_l

        
            outs.append(h_forward_l)
            outs_rev.append(h_back_l)

        output = []
        
        # Take only last time step. Modify for seq to seq
        for t in range(input.shape[0]):
            if merge_mode == 'concat':
                out = outs[-(t + 1)].squeeze()
                out_rev = outs_rev[t].squeeze()
                output.append(torch.cat((out, out_rev), 1))
            elif merge_mode == 'sum':
                output.append(outs[-(t + 1)] + outs_rev[t])
            elif merge_mode == 'ave':
                output.append((outs[-(t + 1)] + outs_rev[t]) / 2)
            elif merge_mode == 'mul':
                output.append(outs[-(t + 1)] * outs_rev[t])
            elif merge_mode is None:
                output.append([outs[-(t + 1)], outs_rev[t]])

        if torch.cuda.is_available():
            output = torch.tensor([item.detach().numpy() for item in output]).cuda()
        else:
            output = torch.tensor([item.detach().numpy() for item in output])
        '''
        
        output = torch.tensor([item.detach().numpy() for item in outs])
        '''
        '''
        out = outs[-1].squeeze()
        out_rev = outs_rev[0].squeeze()
        out = torch.cat((out, out_rev), 1)
        '''

        for layer in range(self.num_layers):
            h0[layer] = torch.tensor([item.detach().numpy() for item in hidden_forward[layer]])
            h0[self.num_layers + layer] = torch.tensor([item.detach().numpy() for item in hidden_backward[layer]])
                
        
        #out = self.fc(out)
        return output, h0

    '''
    def parameters(self, recurse: bool = True):
        for i in self.run_list:
            for name, param in self.rnn_cell_list[i].named_parameters(recurse=recurse):
                yield param
    '''
    
        
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
    def __init__(self, hidden_size, num_words, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.num_words = num_words
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = BidirRecurrentModel("GRU", hidden_size, hidden_size, n_layers, num_words, embedding)
        #self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        # Convert word indexes to embeddings
        #embedded = self.embedding(input_seq)
        outputs, hidden = self.gru(input_seq, hidden, 'concat')
        
        '''
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        '''
        
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden

    def parameters(self, recurse: bool = True):
        for i in self.gru.run_list:
            for name, param in self.gru.rnn_cell_list[i].named_parameters(recurse=recurse):
                yield param

    ######################################################################
# Define Decoder’s Attention Module
# -------------------------------
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



######################################################################
# Define Training Procedure
# -------------------------
#
# Masked loss
# ~~~~~~~~~~~
#
# Since we are dealing with batches of padded sequences, we cannot simply
# consider all elements of the tensor when calculating loss. We define
# ``maskNLLLoss`` to calculate our loss based on our decoder’s output
# tensor, the target tensor, and a binary mask tensor describing the
# padding of the target tensor. This loss function calculates the average
# negative log likelihood of the elements that correspond to a *1* in the
# mask tensor.
#

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


######################################################################
# Single training iteration
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``train`` function contains the algorithm for a single training
# iteration (a single batch of inputs).
#
# We will use a couple of clever tricks to aid in convergence:
#
# -  The first trick is using **teacher forcing**. This means that at some
#    probability, set by ``teacher_forcing_ratio``, we use the current
#    target word as the decoder’s next input rather than using the
#    decoder’s current guess. This technique acts as training wheels for
#    the decoder, aiding in more efficient training. However, teacher
#    forcing can lead to model instability during inference, as the
#    decoder may not have a sufficient chance to truly craft its own
#    output sequences during training. Thus, we must be mindful of how we
#    are setting the ``teacher_forcing_ratio``, and not be fooled by fast
#    convergence.
#
# -  The second trick that we implement is **gradient clipping**. This is
#    a commonly used technique for countering the “exploding gradient”
#    problem. In essence, by clipping or thresholding gradients to a
#    maximum value, we prevent the gradients from growing exponentially
#    and either overflow (NaN), or overshoot steep cliffs in the cost
#    function.
#
# .. figure:: /_static/img/chatbot/grad_clip.png
#    :align: center
#    :width: 60%
#    :alt: grad_clip
#
# Image source: Goodfellow et al. *Deep Learning*. 2016. https://www.deeplearningbook.org/
#
# **Sequence of Operations:**
#
#    1) Forward pass entire input batch through encoder.
#    2) Initialize decoder inputs as SOS_token, and hidden state as the encoder's final hidden state.
#    3) Forward input batch sequence through decoder one time step at a time.
#    4) If teacher forcing: set next decoder input as the current target; else: set next decoder input as current decoder output.
#    5) Calculate and accumulate loss.
#    6) Perform backpropagation.
#    7) Clip gradients.
#    8) Update encoder and decoder model parameters.
#
#
# .. Note ::
#
#   PyTorch’s RNN modules (``RNN``, ``LSTM``, ``GRU``) can be used like any
#   other non-recurrent layers by simply passing them the entire input
#   sequence (or batch of sequences). We use the ``GRU`` layer like this in
#   the ``encoder``. The reality is that under the hood, there is an
#   iterative process looping over each time step calculating hidden states.
#   Alternatively, you can run these modules one time-step at a time. In
#   this case, we manually loop over the sequences during the training
#   process like we must do for the ``decoder`` model. As long as you
#   maintain the correct conceptual model of these modules, implementing
#   sequential models can be very straightforward.
#
#


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, batch_size, clip, max_length=MAX_LENGTH):
    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    

        # Initialize optimizers
    #print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)


    # If you have cuda, configure cuda to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    # Perform backpropatation
    loss.backward()

    
    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


######################################################################
# Training iterations
# ~~~~~~~~~~~~~~~~~~~
#
# It is finally time to tie the full training procedure together with the
# data. The ``trainIters`` function is responsible for running
# ``n_iterations`` of training given the passed models, optimizers, data,
# etc. This function is quite self explanatory, as we have done the heavy
# lifting with the ``train`` function.
#
# One thing to note is that when we save our model, we save a tarball
# containing the encoder and decoder state_dicts (parameters), the
# optimizers’ state_dicts, the loss, the iteration, etc. Saving the model
# in this way will give us the ultimate flexibility with the checkpoint.
# After loading a checkpoint, we will be able to use the model parameters
# to run inference, or we can continue training right where we left off.
#

def trainIters(model_name, voc, pairs, encoder, decoder, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0


######################################################################
# Run Model
# ---------
#
# Finally, it is time to run our model!
#
# Regardless of whether we want to train or test the chatbot model, we
# must initialize the individual encoder and decoder models. In the
# following block, we set our desired configurations, choose to start from
# scratch or set a checkpoint to load from, and build and initialize the
# models. Feel free to play with different model configurations to
# optimize performance.
#

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
encode_hidden_size = 500
decode_hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, decode_hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(encode_hidden_size, voc.num_words, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, decode_hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


######################################################################
# Run Training
# ~~~~~~~~~~~~
#
# Run the following block if you want to train the model.
#
# First we set training parameters, then we initialize our optimizers, and
# finally we call the ``trainIters`` function to run our training
# iterations.
#

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
#n_iteration = 1  #just used for test
print_every = 10
save_every = 4000

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()



# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)

