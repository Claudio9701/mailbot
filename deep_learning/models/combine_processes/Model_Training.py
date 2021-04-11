#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pickle as pk
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow.keras.preprocessing import sequence
from nltk.translate.bleu_score import sentence_bleu


# In[2]:


# Conversational Model Metric
def perplexity(y_true, y_pred):
    return pow(2, categorical_crossentropy(y_pred, y_true))

def bleu(y_true, y_pred):
    y_true = np.array(y_true).tolist()
    y_pred = np.array(y_pred).tolist()
    return sentence_bleu(y_true, y_pred)


# In[3]:


def createModelDirs(output_dir, model_dir, c=0):
    '''Return model directories for outputing logs and checkpoints'''
    final_dir = os.path.join(output_dir,model_dir+'_v{}'.format(c))
    chpts_dir = os.path.join(final_dir,'chpts')
    logs_dir = os.path.join(final_dir,'logs')  
    
    if model_dir+'_v{}'.format(c) in os.listdir(output_dir):
        c += 1
        final_dir, chpts_dir, logs_dir = createModelDirs(output_dir, model_dir, c)
    else:
        os.mkdir(final_dir)
        os.mkdir(chpts_dir)
        os.mkdir(logs_dir)
    
    return final_dir, chpts_dir, logs_dir


# In[4]:


# load variables
word_freqs_inp = pk.load(open('output/data_cleaning_nlp/word_freqs_input.pk', 'rb'))
word_freqs_out = pk.load(open('output/data_cleaning_nlp/word_freqs_output.pk', 'rb'))
x = pk.load(open('output/data_cleaning_nlp/input_data.pk', 'rb'))
y = pk.load(open('output/data_cleaning_nlp/target_data.pk', 'rb'))


# In[5]:


## Hyper-parameters

# data features 
MAX_FEATURES_input = 1000
input_len = 125
MAX_FEATURES_output = 1000
target_len = 125

# training parameters
num_epochs = 5
batch_size = 32

# model estructure
embed_size = 512
hidden_size = 264
n_encoder_layers = 3
encoder_hidden_sizes = [256, 128, 64]
n_decoder_layers = 3
lstm_hidden_sizes = [256, 128, 64]


# In[6]:


# Define output dir
outDir = 'output/'
actualDir = 'trained_model'

print()
if not(actualDir in os.listdir(outDir)):
    os.mkdir(os.path.join(outDir, actualDir))
    print('output dir created')
else:
    print('output dir already created')
print()


# In[7]:


# Define directories for outputs
actual_outDir = os.path.join(outDir, actualDir)
modelDir = 'model_epochs-{}_batch-{}_hidden-{}_embed-{}'.format(num_epochs, batch_size, hidden_size, embed_size)
finalDir, chptsDir, logsDir = createModelDirs(actual_outDir,modelDir)


# In[8]:


### Build vocabulary of unique words 

## Inputs
vocab_size_input = min(MAX_FEATURES_input, len(word_freqs_inp)) + 2
word2index_inp = {x[0]: i+2 for i, x in enumerate(word_freqs_inp.most_common(MAX_FEATURES_input))}
word2index_inp["PAD"] = 0
word2index_inp["UNK"] = 1
index2word_inp = {v:k for k, v in word2index_inp.items()}

## Outputs
vocab_size_output = min(MAX_FEATURES_output, len(word_freqs_out)) + 4
word2index_out = {x[0]: i+4 for i, x in enumerate(word_freqs_out.most_common(MAX_FEATURES_output))}
word2index_out["PAD"] = 0
word2index_out["UNK"] = 1
word2index_out["GO"] = 2
word2index_out["EOS"] = 3
index2word_out = {v:k for k, v in word2index_out.items()}

# Save dictionaries in model directory
pk.dump(word2index_inp, open(os.path.join(finalDir,'word2index_inp.pk'),'wb'))
pk.dump(index2word_inp, open(os.path.join(finalDir,'index2word_inp.pk'),'wb'))
pk.dump(word2index_out, open(os.path.join(finalDir,'word2index_out.pk'),'wb'))
pk.dump(index2word_out, open(os.path.join(finalDir,'index2word_out.pk'),'wb'))


# In[9]:


# Filter records by lenght 
x_new = []
y_new = []

for input_, target_ in zip(x,y):
    if all([len(input_) <= input_len, len(input_) > 0, len(target_) <= target_len, len(target_) > 0]):
        x_new.append(input_)
        y_new.append(target_)
        
print('number of records after filtering by lenght:', len(x_new))

# Create a copy of conversations with the words replaced by their IDs
X_input = np.empty((len(x_new),), dtype=list)
y_input = np.empty((len(y_new),), dtype=list)
y_target_ids = np.empty((len(y_new),), dtype=list)

for i in range(len(x_new)):
    seqs_x = []
    seqs_y_input = []
    seqs_y_target = []
    
    # Replace input sequences IDs
    for word in x_new[i]:
        if word in word2index_inp:
            seqs_x.append(word2index_inp[word])
        else:
            seqs_x.append(word2index_inp["UNK"]) # Replace words with low frequency with <UNK>
               
    # Target sequences IDs
    seqs_y_input = [word2index_out["GO"]] # Start of Sentence ID
    for word in y_new[i]:
        if word in word2index_out:
            seqs_y_input.append(word2index_out[word])
            seqs_y_target.append(word2index_out[word])
        else:
            # Replace words with low frequency with <UNK>
            seqs_y_input.append(word2index_out["UNK"])
            seqs_y_target.append(word2index_out["UNK"])
    seqs_y_target.append(word2index_out["EOS"]) # End of Sentece ID

    X_input[i] = seqs_x
    y_input[i] = seqs_y_input
    y_target_ids[i] = seqs_y_target

X_input = sequence.pad_sequences(X_input, input_len, padding='post')
y_input = sequence.pad_sequences(y_input, target_len, padding='post')
y_target_ids = sequence.pad_sequences(y_target_ids, target_len, padding='post')

# Create one-hot target variable
y_target = np.empty((len(y_target_ids), target_len, vocab_size_output))

for i in range(len(y_target_ids)):
    for j in range(target_len):
        y_target[i, j, y_target_ids[i,j]] = 1
        
print("y_target size = %f gigabytes" % ((y_target.size * y_target.itemsize)/1e9))

# Save X and y input
pk.dump(X_input, open(os.path.join(finalDir,'x_inp.pk'),'wb'))
pk.dump(y_input, open(os.path.join(finalDir,'y_inp.pk'),'wb'))


# In[11]:


## Tensorflow Keras Conversational Model 
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))

# Set up encoder, output lstm states
encoder_embed_layer = Embedding(vocab_size_input, embed_size, mask_zero=True)
encoder_embed = encoder_embed_layer(encoder_inputs)

encoder_layers = [LSTM(encoder_hidden_sizes[i], return_sequences=True, go_backwards=True) for i in range(n_encoder_layers)]
encoder_lstms_outputs = []
for i in range(n_decoder_layers):
    if i == 0:
        encoder_lstms_outputs.append(encoder_layers[i](encoder_embed))
    else:
        encoder_lstms_outputs.append(encoder_layers[i](encoder_lstms_outputs[i-1]))

encoder_lstm, state_h, state_c = LSTM(hidden_size, return_state=True,
                                      go_backwards=True)(encoder_lstms_outputs[-1])
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

decoder_embed_layer = Embedding(vocab_size_output, embed_size, mask_zero=True)
decoder_embed = decoder_embed_layer(decoder_inputs)

decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)

decoder_layers = [LSTM(lstm_hidden_sizes[i], return_sequences=True) for i in range(n_decoder_layers)]
decoder_lstms_outputs = []
for i in range(n_decoder_layers):
    if i == 0:
        decoder_lstms_outputs.append(decoder_layers[i](decoder_outputs))
    else:
        decoder_lstms_outputs.append(decoder_layers[i](decoder_lstms_outputs[i-1]))

# Create dense vector with next word probability 
decoder_dense = Dense(vocab_size_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_lstms_outputs[-1])

# Define the model that will turn 'X_input' and 'y_input' into 'y_target'
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# Compile model
model.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=[perplexity])
# Model Estructure Summary
model.summary()


# In[12]:


## Inference Model
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

decoder_layers = [LSTM(lstm_hidden_sizes[i], return_sequences=True) for i in range(n_decoder_layers)]
decoder_lstms_outputs = []
for i in range(n_decoder_layers):
    if i == 0:
        decoder_lstms_outputs.append(decoder_layers[i](decoder_outputs))
    else:
        decoder_lstms_outputs.append(decoder_layers[i](decoder_lstms_outputs[i-1]))

decoder_outputs = decoder_dense(decoder_lstms_outputs[-1])
decoder_model = Model(
    [decoder_inputs] + decoder_states_input,
    [decoder_outputs] + decoder_states)

# Save models
encoder_model.save(os.path.join(finalDir,'encoder_model_{}_{}_{}_{}.h5'.format(hidden_size,batch_size,num_epochs,embed_size)))
decoder_model.save(os.path.join(finalDir,'decoder_model_{}_{}_{}_{}.h5'.format(hidden_size,batch_size,num_epochs,embed_size)))


# In[13]:


# Define callbacks
model_checkpoint = ModelCheckpoint(os.path.join(chptsDir,'{epoch:02d}_{val_loss:.2f}.chpt'),
                                   monitor='val_loss', verbose=0, save_best_only=True,
                                   save_weights_only=False, mode='auto', period=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=50)

tensorboard = TensorBoard(log_dir=logsDir, histogram_freq=20, batch_size=32, write_graph=True,
                          write_grads=True, embeddings_freq=0, embeddings_layer_names=None,
                          embeddings_metadata=None, embeddings_data=None)


# In[14]:


# Fit model
model_history = model.fit([X_input, y_input], y_target,
                          batch_size=batch_size,
                          epochs=num_epochs,
                          validation_split=0.05,
                          callbacks=[early_stopping, model_checkpoint, tensorboard])


# In[ ]:


# Save model history
with open(os.path.join(finalDir,'history_{}_{}_{}_{}.pk'.format(hidden_size,batch_size,num_epochs,embed_size)),'wb') as f:
    pk.dump(model_history.history, f)


# lemmatizacion: 
#     spacy
#     clips: pattern.es
#         
# 
# diccionarios:
#     lista de palabras
#     
#     
# word vectors:
#     vert
