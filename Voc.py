######################################################################
# Model Overview
# --------------
#
# As mentioned, the model that we are using is a
# `sequence-to-sequence <https://arxiv.org/abs/1409.3215>`__ (seq2seq)
# model. This type of model is used in cases when our input is a
# variable-length sequence, and our output is also a variable length
# sequence that is not necessarily a one-to-one mapping of the input. A
# seq2seq model is comprised of two recurrent neural networks (RNNs) that
# work cooperatively: an **encoder** and a **decoder**.
#
# .. figure:: /_static/img/chatbot/seq2seq_ts.png
#    :align: center
#    :alt: model
#
#
# Image source:
# https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/
#
# Encoder
# ~~~~~~~
#
# The encoder RNN iterates through the input sentence one token
# (e.g. word) at a time, at each time step outputting an “output” vector
# and a “hidden state” vector. The hidden state vector is then passed to
# the next time step, while the output vector is recorded. The encoder
# transforms the context it saw at each point in the sequence into a set
# of points in a high-dimensional space, which the decoder will use to
# generate a meaningful output for the given task.
#
# Decoder
# ~~~~~~~
#
# The decoder RNN generates the response sentence in a token-by-token
# fashion. It uses the encoder’s context vectors, and internal hidden
# states to generate the next word in the sequence. It continues
# generating words until it outputs an *EOS_token*, representing the end
# of the sentence. We use an `attention
# mechanism <https://arxiv.org/abs/1409.0473>`__ in our decoder to help it
# to “pay attention” to certain parts of the input when generating the
# output. For our model, we implement `Luong et
# al. <https://arxiv.org/abs/1508.04025>`__\ ’s “Global attention” module,
# and use it as a submodule in our decode model.
#


######################################################################
# Data Handling
# -------------
#
# Although our models conceptually deal with sequences of tokens, in
# reality, they deal with numbers like all machine learning models do. In
# this case, every word in the model’s vocabulary, which was established
# before training, is mapped to an integer index. We use a ``Voc`` object
# to contain the mappings from word to index, as well as the total number
# of words in the vocabulary. We will load the object later before we run
# the model.
#
# Also, in order for us to be able to run evaluations, we must provide a
# tool for processing our string inputs. The ``normalizeString`` function
# converts all characters in a string to lowercase and removes all
# non-letter characters. The ``indexesFromSentence`` function takes a
# sentence of words and returns the corresponding sequence of word
# indexes.
#

import re
from config import *

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens
        for word in keep_words:
            self.addWord(word)

def preprocess(text):
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = re.sub(r"  "," ",text)

    return text

# Lowercase and remove non-letter characters
def normalizeString(s):
    s = s.lower()
    s = preprocess(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# Takes string sentence, returns sentence of word indexes
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]