import pandas as pd
import numpy as np
import string
numpy==1.20.1
streamlit==0.76.0
seaborn==0.11.1
matplotlib==3.4.1
plotly==4.14.3
pandas==1.2.3
joblib==0.16.0
scikit-learn
from string import digits
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# tenserflow as tf
#from tensorflow.keras.optimizers.legacy import Adam
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from google.colab import drive
drive.mount("/content/gdrive")

#text = open('/content/gdrive/My Drive/Colab Notebooks/mar.txt', 'r').read()

lines = pd.read_table('/content/gdrive/My Drive/Colab Notebooks/mar.txt', names=['eng', 'mar'])
print(lines.shape)

#lower case
lines.eng = lines.eng.apply(lambda x: x.lower())
lines.mar = lines.mar.apply(lambda x: x.lower())

# Remove quotes
lines.eng=lines.eng.apply(lambda x: re.sub("'", '', x))
lines.mar=lines.mar.apply(lambda x: re.sub("'", '', x))

exclude = set(string.punctuation) # Set of all special characters
# Remove all the special characters
lines.eng=lines.eng.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines.mar=lines.mar.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

# Remove all numbers from text
remove_digits = str.maketrans('', '', digits)
lines.eng=lines.eng.apply(lambda x: x.translate(remove_digits))
lines.mar = lines.mar.apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

# Remove extra spaces
lines.eng=lines.eng.apply(lambda x: x.strip())
lines.mar=lines.mar.apply(lambda x: x.strip())
lines.eng=lines.eng.apply(lambda x: re.sub(" +", " ", x))
lines.mar=lines.mar.apply(lambda x: re.sub(" +", " ", x))

# Add start and end tokens to target sequences
lines.mar = lines.mar.apply(lambda x : 'START_ '+ x + ' _END')

print(lines.sample(10))

# Vocabulary of English
all_eng_words=set()
for eng in lines.eng:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

# Vocabulary of French 
all_marathi_words=set()
for mar in lines.mar:
    for word in mar.split():
        if word not in all_marathi_words:
            all_marathi_words.add(word)
# Max Length of source sequence
lenght_list=[]
for l in lines.eng:
    lenght_list.append(len(l.split(' ')))
max_length_src = np.max(lenght_list)
print()
print(max_length_src)

# Max Length of target sequence
lenght_list=[]
for l in lines.mar:
    lenght_list.append(len(l.split(' ')))
max_length_tar = np.max(lenght_list)
print()
print(max_length_tar)

input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_marathi_words))
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_marathi_words)
print(num_encoder_tokens, num_decoder_tokens)

num_decoder_tokens += 1 # For zero padding
print('num decoder tokens')
print(num_decoder_tokens)

input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])
reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())
lines = shuffle(lines)
print(lines.head(10))

# Train - Test Split
X, y = lines.eng, lines.mar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
print('train test')
print(len(X_train), X_test.shape)

X_train.to_pickle('/content/gdrive/My Drive/Colab Notebooks/Weights_Mar/X_train.pkl')
X_test.to_pickle('/content/gdrive/My Drive/Colab Notebooks/Weights_Mar/X_test.pkl')
def generate_batch(X = X_train, y = y_train, batch_size = 128):
    ''' Generate a batch of data '''
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = np.zeros((batch_size, max_length_src),dtype='float32')
            decoder_input_data = np.zeros((batch_size, max_length_tar),dtype='float32')
            decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text.split()):
                    encoder_input_data[i, t] = input_token_index[word] # encoder input seq
                for t, word in enumerate(target_text.split()):
                    if t<len(target_text.split())-1:
                        decoder_input_data[i, t] = target_token_index[word] # decoder input seq
                    if t>0:
                        # decoder target sequence (one hot encoded)
                        # does not include the START_ token
                        # Offset by one timestep
                        decoder_target_data[i, t - 1, target_token_index[word]] = 1.
            yield([encoder_input_data, decoder_input_data], decoder_target_data)

latent_dim = 50

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb =  Embedding(num_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
dec_emb = dec_emb_layer(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

#from IPython.display import Image
#Image(retina=True, filename='train_model.png')

train_samples = len(X_train)
val_samples = len(X_test)
batch_size = 128
epochs = 15
model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size),
                    steps_per_epoch = train_samples//batch_size,
                    epochs=epochs,
                    validation_data = generate_batch(X_test, y_test, batch_size = batch_size),
                    validation_steps = val_samples//batch_size)

model.save_weights('nmt_weights.h5')

model.load_weights('nmt_weights.h5')

# Encode the input sequence to get the "thought vectors"
encoder_model = Model(encoder_inputs, encoder_states)

