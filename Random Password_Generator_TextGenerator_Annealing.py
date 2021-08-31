import pandas as pd
import numpy as np
import sys
import random

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


def get_optimizer():
    return Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def prepare_data(password_len=50, input_len=5, step=2):
    max_len = password_len
    
    Password_data_path = 'Dataset/Dangerous_password/data.csv'
    Password_data = pd.read_csv(Password_data_path, error_bad_lines=False)
    full_password_string = []
    text = ''
    
    print(Password_data.columns)
    
    for i in range(0, len(Password_data[Password_data.columns[0]])):
        if Password_data['strength'][i] == 2:
            kalan_uzunluk = max_len - len(str(Password_data['password'][i]))
            bosluk = ''
            for j in range(0, kalan_uzunluk):
                bosluk = bosluk + ' '
            full_password = str(Password_data['password'][i]) + bosluk
            full_password_string.append(full_password)
            text = text + full_password
    
    print('Password_data_array.shape: ', np.array(full_password_string).shape)
    
    
    input_len = input_len
    step = step
    sentences = []
    next_chars = []
    
    for i in range(0, len(full_password_string)):
        for j in range(0, max_len - input_len, step):
            sentences.append(full_password_string[i][j:j+input_len])
            next_chars.append(full_password_string[i][j+input_len])
            if full_password_string[i][j] == ' ':
                break
    
    print('Number of sequences:', len(sentences))
    
    chars = sorted(list(set(text))) ## 130
    #print(chars)
    print('Unique characters:', len(chars))
    char_indices = dict((char, chars.index(char)) for char in chars)
    
    print('Vectorization...')
    x = np.zeros((len(sentences), input_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
                y[i, char_indices[next_chars[i]]] = 1
    
    return x, y, chars, char_indices, full_password_string
            

def get_model(chars, input_len=5):
    model = Sequential()
    model.add(layers.LSTM(128, input_shape=(input_len, len(chars))))
    model.add(layers.Dense(len(chars), activation='softmax'))
    
    adam = get_optimizer()
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    return model


def train_model(model, password_list, chars, char_indices, x, y, epoch=100, input_len=5, password_len=50):
    max_len=password_len
    for epoch in range(1, epoch):
        print('epoch', epoch)
        model.fit(x, y, batch_size=128, epochs=1)
        
        start_index = random.randint(0, len(password_list) - 1)
        generated_text = password_list[start_index][0:input_len]
        
        print('--- Generating with seed: "' + generated_text + '"')
        
        for temperature in [0.2, 0.5, 1.0, 1.2, 1.5]:
            ###soğuma 0.95 temperature = temperature * 0.95
            
            print('------ temperature:', temperature)
            sys.stdout.write(generated_text)
            temp_generated_text = generated_text
            for i in range(0, max_len - input_len):
                sampled = np.zeros((1, input_len, len(chars)))
                for t, char in enumerate(temp_generated_text):
                    sampled[0, t, char_indices[char]] = 1.
                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_char = chars[next_index]
                temp_generated_text += next_char
                temp_generated_text = temp_generated_text[1:]
                sys.stdout.write(next_char)
                temperature = temperature * 0.95
            sys.stdout.write('\n')
    filename = 'Generator_models/LSTM_model.h5'
    model.save_weights(filename)
        

if __name__ == '__main__':
    x, y, chars, char_indices, password_list = prepare_data(password_len=50, input_len=5, step=2)
    model = get_model(chars, input_len=5)
    train_model(model, password_list, chars, char_indices, x, y, epoch=100, input_len=5, password_len=50)
    






