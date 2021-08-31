import numpy as np
import pickle
import pandas as pd
import random

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


max_len = 50
input_len = 5


def get_optimizer():
    return Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


def get_model(chars_len, input_len=5):
    model = Sequential()
    model.add(layers.LSTM(128, input_shape=(input_len, chars_len)))
    model.add(layers.Dense(chars_len, activation='softmax'))
    
    adam = get_optimizer()
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    return model


def password_classification(clf_model, password):
        kalan_uzunluk = max_len - len(str(password))
        bosluk = ''
        for j in range(0, kalan_uzunluk):
            bosluk = bosluk + ' '
        full_password = str(password) +  bosluk
        
        for j in range(0, max_len):
            password_array[0][j] = ord(full_password[j])
        
        prediction_strength = clf_model.predict(password_array)
        
        print('Your password : ', password)
        print('Your password strength : ', prediction_strength)


def prepare_data(password_len=50, input_len=5, step=2):
    max_len = password_len
    
    Password_data_path = 'Dataset/Dangerous_password/data.csv'
    Password_data = pd.read_csv(Password_data_path, error_bad_lines=False, warn_bad_lines=False)
    full_password_string = []
    text = ''
    
    for i in range(0, len(Password_data[Password_data.columns[0]])):
        if Password_data['strength'][i] == 2:
            kalan_uzunluk = max_len - len(str(Password_data['password'][i]))
            bosluk = ''
            for j in range(0, kalan_uzunluk):
                bosluk = bosluk + ' '
            full_password = str(Password_data['password'][i]) + bosluk
            full_password_string.append(full_password)
            text = text + full_password
    
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
    
    chars = sorted(list(set(text))) ## 130
    char_indices = dict((char, chars.index(char)) for char in chars)
    
    return chars, char_indices, full_password_string


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def password_generation(generator_model, chars, char_indices, password_list, temperature=1.2):
    temperature = temperature
    
    start_index = random.randint(0, len(password_list) - 1)
    generated_text = password_list[start_index][0:input_len]
    
    temp_generated_text = generated_text
    for i in range(0, max_len - input_len):
        sampled = np.zeros((1, input_len, len(chars)))
        for t, char in enumerate(temp_generated_text):
            sampled[0, t, char_indices[char]] = 1.
        preds = generator_model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = chars[next_index]
        generated_text += next_char
        
        temp_generated_text += next_char
        temp_generated_text = temp_generated_text[1:]
        temperature = temperature * 0.95
    
    return generated_text


print('strength -1: this password was used for brute force attack')
print('strength 0: this password is weak')
print('strength 1: this password is medium weakness')
print('strength 2: this password is strong\n')
print('Press q to exit')
print('Press r for random password\n')
if __name__ == "__main__":
    print('Data Initializing ...')
    chars, char_indices, password_list = prepare_data(password_len=max_len, input_len=input_len, step=2)
    
    clf_model = pickle.load(open('Clf/rf.sav', 'rb'))
    generator_model = get_model(len(chars), input_len=input_len)
    generator_model.load_weights("Generator_models/Text_generator_LSTM/LSTM_model.h5")
    password_array = np.zeros((1, max_len))
    
    while(True):
        password = input("Enter your password : ")
        
        if password == 'q':
            break
        elif password == 'r':
            password_generated = password_generation(generator_model, chars, char_indices,
                                           password_list, temperature=1.2)
            password_classification(clf_model, password_generated)
        else:
            password_classification(clf_model, password)
        
       






