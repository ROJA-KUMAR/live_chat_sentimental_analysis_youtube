from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model('dl.h5')

# Load the label encoder and tokenizer
data = pd.read_csv('live_chat_sentiments5.csv')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Message'])
label_encoder = pd.read_csv('label_encoder_classes.csv')['0'].to_list()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']  # Match the input field name in your HTML form

        sequence_length = 28 
        # Preprocess user input and predict emotion
        text = user_input.lower().replace('[^\w\s]', '')

        # Tokenize and pad the text
        sequence = tokenizer.texts_to_sequences([text])
        sequence = pad_sequences(sequence, maxlen=sequence_length)  # Replace 100 with the desired sequence length

        prediction = model.predict(sequence)
        predicted_emotion = label_encoder[np.argmax(prediction)]

        return render_template('index.html', user_input=user_input, predicted_emotion=predicted_emotion)

if __name__ == '__main__':

    app.run(debug=True)
