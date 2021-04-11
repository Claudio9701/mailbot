#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model
from keras.models import Sequential, Input, Model
from keras.layers import LSTM, Dense, Embedding
from keras.optimizers import SGD, Adam, Nadam

word2index = pickle.load(open('w2i.pk','rb'))
index2word = pickle.load(open('i2w.pk','rb'))

X_input = np.load('data/input/X_input.npy')
y_input = np.load('data/input/y_input.npy')
y_target = np.load('data/input/y_target.npz')['arr_0']

# nlp parameters
vocab_size = 502

# model parameters
num_epochs = 100
batch_size = 32
hidden_size = 128
embed_size = 256

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
encoder_embed_layer = Embedding(vocab_size, embed_size)
encoder_embed = encoder_embed_layer(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(hidden_size, return_state=True)(encoder_embed)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

decoder_embed_layer = Embedding(vocab_size, hidden_size)
decoder_embed = decoder_embed_layer(decoder_inputs)

decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)

decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# ```python
# %%time
#
# parallel_model = multi_gpu_model(model, gpus=2)
# parallel_model.summary()
#
# # Compile & run training
# parallel_model.compile(optimizer=Nadam(), loss='categorical_crossentropy')
#
# # Note that `decoder_target_data` needs to be one-hot encoded,
# # rather than sequences of integers like `decoder_input_data`!
# parallel_model_history = parallel_model.fit([X_input, y_input], y_target,
#                                              batch_size=batch_size,
#                                              epochs=num_epochs,
#                                              validation_split=0.2)
# ```

# Compile & run training
model.compile(optimizer=Nadam(), loss='categorical_crossentropy')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
filepath = 'model_{epoch:02d}_{val_loss:.2f}.chpt'
model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                   save_weights_only=False, mode='auto', period=1)

# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model_history = model.fit([X_input, y_input], y_target,
                          batch_size=batch_size,
                          epochs=num_epochs,
                          validation_split=0.2,
                          callbacks=[early_stopping, model_checkpoint])

model.save('model_{}_{}_{}_{}.h5'.format(hidden_size,batch_size,num_epochs,embed_size))

%matplotlib inline
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'], 'g')
plt.show()

with open('history_{}_{}_{}_{}.pk'.format(hidden_size,batch_size,num_epochs,embed_size),'wb') as f:
    pickle.dump(model_history, f)

# Inference Model
# Encoder
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder inference inputs
decoder_state_input_h = Input(shape=(hidden_size,))
decoder_state_input_c = Input(shape=(hidden_size,))
decoder_states_input = [decoder_state_input_h, decoder_state_input_c]

# Decoder inference
decoder_embed = decoder_embed_layer(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embed, initial_state=decoder_states_input)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_input,
    [decoder_outputs] + decoder_states)

# nlp parameters
word2index['START'] = 501
max_seq_len = 200

def decode_sequence(input_seq):
    '''Generates answer from given sequence of word indexes'''
    states_value = encoder_model.predict(input_seq)

    #target_seq = np.zeros((1, 1, vocab_size))
    #target_seq[0, 0, word2index['START']] = 1.

    stop_condition = False
    decode_sentence = ''

    while not stop_condition:
        output_word_index, h, c = decoder_model.predict(
            [input_seq] + states_value)

        sampled_word_index = np.argmax(output_word_index[0, -1, :])
        sampled_word = index2word[sampled_word_index]
        decode_sentence += sampled_word

        if (sampled_word == 'END' or
            len(decode_sentence) > max_seq_len):
            stop_condition = True
        else:
            print(decode_sentence)

        target_seq = np.zeros((1,1,vocab_size))
        target_seq[0, 0, sampled_word_index] = 1.

        states_value = [h, c]

    return decode_sentence
