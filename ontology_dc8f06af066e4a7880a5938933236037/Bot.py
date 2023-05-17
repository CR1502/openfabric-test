import nltk
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Bidirectional, LSTM, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Read the dataset from CSV file
dataset = pd.read_csv('scraped_data.csv')

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset['Title'].values)
vocab_size = len(tokenizer.word_index) + 1

# Convert questions to sequences of integers
sequences = tokenizer.texts_to_sequences(dataset['Title'].values)

# Pad sequences to ensure uniform length
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Prepare input-output pairs
X = padded_sequences
Y = pd.get_dummies(dataset['Content']).values

model = Sequential()
model.add(Embedding(vocab_size, 516, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(516, kernel_regularizer=regularizers.l2(0.01), return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(GRU(128, kernel_regularizer=regularizers.l2(0.01))))
model.add(Dense(128, activation='elu'))
model.add(Dense(64, activation='gelu'))
model.add(Dense(Y.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
history = model.fit(X, Y, epochs=1500, batch_size=32, verbose=1)

# Accuracy chart
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Save the trained model to an HDF5 file
model.save('chatbot_model.h5')


# Tokenize and vectorize user's question
def tokenize_vectorize(question):
    question_tokens = word_tokenize(question)
    question_sequence = tokenizer.texts_to_sequences([question_tokens])
    question_input = pad_sequences(question_sequence, maxlen=max_sequence_length)
    return question_input


# Extract keywords from user's question
def extract_keywords(question):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(question)
    keywords = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stop_words]
    return keywords


# Define a function to generate a response
def generate_response(question):
    # Tokenize and vectorize the question
    question_input = tokenize_vectorize(question)

    # Get the predicted answer
    predicted_answer = model.predict(question_input)

    # Get the index of the predicted answer
    answer_index = np.argmax(predicted_answer)

    # Get the actual answer from the dataset
    response = dataset['Content'].values[answer_index]

    return response


# Interactive prompt
while True:
    user_question = input("User: ")
    if user_question.lower() == 'exit':
        break

    # Extract keywords from user's question
    keywords = extract_keywords(user_question)
    print("Keywords:", keywords)

    bot_answer = generate_response(user_question)
    print("ChatBot:", bot_answer)
