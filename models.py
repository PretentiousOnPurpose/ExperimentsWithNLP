import numpy as np
import keras
import tensorflow as tf

def LSTM_NMT(embedded_dim, input_dim, num_layers=1):
    encoder_inputs = keras.Input(shape=(None, input_dim))
    encoder = keras.layers.LSTM(embedded_dim, return_state=True)

    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    encoder_states = [state_h, state_c]

    decoder_inputs = keras.Input(shape=(None, input_dim))

    decoder_lstm = keras.layers.LSTM(embedded_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(input_dim, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)   

    return model