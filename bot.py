import nltk

import os
import json
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ChatbotModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128) # self.fc1 is a matrix of weights
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # drops 50% of neurons randomly

    # x is the input
    def forward(self, x):
        x = self.relu(self.fc1(x)) # multiplication between matrix self.fc1 and input x, then applies relu
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x) # output vector

        return x

class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path

        self.documents = [] # examples of user inputs and responses
        self.vocabulary = [] # all unique words the chatbot knows and can process
        self.intents = [] # different categories like "greeting" or "help" that user requests
        self.intents_responses = {} # possible responses to intents
        
        self.function_mappings = function_mappings # connect specifc intents to functions that perform actions such as placing an order or fetching your stock portfolio
        
        self.X = None # X is a matrix for input data
        self.y = None # y is the output vector

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer() # a lemmatizer reduces a word to its base form (ex. ran -> run, better -> good)

        words = nltk.word_tokenize(text) # splits input text into separate words
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]
    
    def parse_intents(self):
        # lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path): # checks if the intents file exists
            with open(self.intents_path, 'r') as f: # The "with" command automatically opens and closes the file. The 'r' means read-only.
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag'])) # here we are appending tuples of words with corresponding tags

                self.vocabulary = sorted(set(self.vocabulary)) # sorted list with unique words

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words) # turns every word into 0s and 1s

            intent_index = self.intents.index(document[1]) # tag

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss() # measures how wrong our model is
        optimizer = optim.Adam(self.model.parameters(), lr=lr) # decides how to adjust the weights to improve

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y) # measures how far off our predictions are from the correct answer batch_y
                loss.backward() # chain rule of neural networks
                optimizer.step() # updates weights to reduce the error, dependant on the learning rate (lr)
                running_loss += loss
            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}") # prints the average loss (smaller numbers means better performance)

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path) # saves trained model weights to a file so we can use them later

        with open(dimensions_path, 'w') as f:
            json.dump({ 'input_size': self.X.shape[1], 'output_size': len(self.intents) }, f) # saves learned information (trained weights, input and output sizes) to a separate JSON file

    def load_model(self, model_path, dimensions_path): # model_path: path to where model weights are stored
    # dimensions_path: path to where model architecture info is stored
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f) # converts data from f into a dictionary dimensions

        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()
        
        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None


def get_stocks():
    stocks = ['APPL', 'META', 'NVDA', 'GS', 'MSFT']

    print(random.sample(stocks, 3))


if __name__ == '__main__':
    # assistant = ChatbotAssistant('intents.json', function_mappings = {'stocks': get_stocks})
    # assistant.parse_intents()
    # assistant.prepare_data()
    # assistant.train_model(batch_size=8, lr=0.001, epochs=100)
    
    # assistant.save_model('chatbot_model.pth', 'dimensions.json')

    assistant = ChatbotAssistant('intents.json', function_mappings = {'stocks': get_stocks})
    assistant.parse_intents()
    assistant.load_model('chatbot_model.pth', 'dimensions.json')

    while True:
        message = input('Enter your message: ')

        if message == '/quit':
            break

        print(assistant.process_message(message))