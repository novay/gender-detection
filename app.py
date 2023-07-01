import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

model = tf.keras.models.load_model(os.getcwd()+'/models/gender-lstm-base.h5')

def name_to_vector(name):
    sequence = name_to_sequence(name)
    vector = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=29, padding='post')[0]
    return vector

def name_to_sequence(name):
    char_to_int = dict((c, i) for i, c in enumerate('abcdefghijklmnopqrstuvwxyz '))
    sequence = [char_to_int[char.lower()] for char in name if char.lower() in char_to_int]
    return sequence

def predict_gender(name):
    name_vector = np.array(name_to_vector(name))
    padded_name_vector = tf.keras.preprocessing.sequence.pad_sequences([name_vector], maxlen=29, padding='post', truncating='post')
    prediction = model.predict(padded_name_vector, verbose=0)
    
    if prediction < 0.5:
        return 'Male'
    else:
        return 'Female'

@app.route('/')
def welcome():
    return 'Selamat datang di API Prediksi Gender menggunakan LSTM!'

@app.route('/prediksi')
def prediksi_gender():
    nama = request.args.get('nama')
    hasil_prediksi = predict_gender(nama)
    
    return jsonify({'gender': hasil_prediksi})

if __name__ == '__main__':
    app.run()