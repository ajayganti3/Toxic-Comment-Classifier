import re
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import initializers, optimizers, layers

from nltk.corpus import stopwords

warnings.simplefilter(action="ignore")

## Function for cleaning the text
def process_text(data):
    stop = stopwords.words('english')
    data['processed_text'] = data.apply(lambda row: row['comment_text'].replace("\n"," "), axis=1) ## Remove new lines
    data['processed_text'] = data.apply(lambda row: re.sub('http://\S+|https://\S+', 'urls',row['processed_text']).lower(), axis=1) # Remove URL's
    data['processed_text'] = data.apply(lambda row: re.sub('[^A-Za-z ]+', '',row['processed_text']).lower(), axis=1) # Removes special characters, punctuations except alphabets
    data['processed_text'] = data['processed_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) # Removes Stop words
    data['processed_text'] = data.apply(lambda row: re.sub('  +', ' ',row['processed_text']).strip(), axis=1) # Removes extra spaces in between the words
    data['processed_text'] = data.apply(lambda x: x['comment_text'] if len(x['processed_text'])==0 else x['processed_text'], axis=1)
    return data

def load_model():
        ## Get the path of the current file
    directory = os.path.dirname(__file__)
    model_filename = os.path.join(directory, 'model','best_model_weights-03.hdf5')
    model = tf.keras.models.load_model(model_filename)

    ## Load the tokenizer
    token_filename = os.path.join(directory, 'model','token.pkl')
    with open(token_filename, 'rb') as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

def predict(text):
    text_lst = []
    text_lst.append(text)
    
    # Load the model
    model, tokenizer = load_model()
    sample_df = pd.DataFrame(data = text_lst, columns = ['comment_text'])
    sample_df = process_text(sample_df)
    test_tokens = tokenizer.texts_to_sequences(sample_df['processed_text'])
    test_seq = pad_sequences(test_tokens, maxlen=300)
    
    # Predict the output and assign labels
    y_pred = model.predict(test_seq)
    labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
    prediction_labels = {}
    for j, pred in enumerate(y_pred):
        temp = []
        for i in range(len(pred)):
            prob = pred[i]
            if prob>=0.5:
                temp.append(labels[i])
        prediction_labels[j+1] = temp

    # Generating the proper formatted output 
    prediction = prediction_labels[1]
    output = "The given comment is "
    if len(prediction) != 0:
        for i,j in enumerate(prediction):
            if i == 0:
                output += j
            elif (i+1) == len(prediction):
                output = output + " and " + j
            else:
                output = output + ", " + j
    else:
        output = "The given comment is not toxic"
    return output