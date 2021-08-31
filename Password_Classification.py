import pandas as pd
import numpy as np
from tqdm import trange

from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle


max_len = 50

def prapera_data_to_text():
    Password_data_path = 'Dataset/Dangerous_password/data.csv'
    
    Password_data = pd.read_csv(Password_data_path, error_bad_lines=False) 
    
    print(Password_data.head())
    print(Password_data.columns)
    print(len(Password_data[Password_data.columns[0]]))
    
    attacked_password = []
    
    file_1 = open("Dataset/Dangerous_password/1.txt","r+", encoding='utf8', errors='ignore')
    file_2 = open("Dataset/Dangerous_password/2.txt","r+", encoding='utf8', errors='ignore')
    file_3 = open("Dataset/Dangerous_password/3.txt","r+", encoding='utf8', errors='ignore')
    file_4 = open("Dataset/Dangerous_password/4.txt","r+", encoding='utf8', errors='ignore')
    file_5 = open("Dataset/Dangerous_password/5.txt","r+", encoding='utf8', errors='ignore')
    file_6 = open("Dataset/Dangerous_password/6.txt","r+", encoding='utf8', errors='ignore')
    file_7 = open("Dataset/Dangerous_password/7.txt","r+", encoding='utf8', errors='ignore')  
    
    file_1_textlist = file_1.readlines()
    file_2_textlist = file_2.readlines()
    file_3_textlist = file_3.readlines()
    file_4_textlist = file_4.readlines()
    file_5_textlist = file_5.readlines()
    file_6_textlist = file_6.readlines()
    file_7_textlist = file_7.readlines()
    
    for i in range(0, len(file_1_textlist)):
        attacked_password.append(file_1_textlist[i])
    for i in range(0, len(file_2_textlist)):
        attacked_password.append(file_2_textlist[i])
    for i in range(0, len(file_3_textlist)):
        attacked_password.append(file_3_textlist[i])
    for i in range(0, len(file_4_textlist)):
        attacked_password.append(file_4_textlist[i])
    for i in range(0, len(file_5_textlist)):
        attacked_password.append(file_5_textlist[i])
    for i in range(0, len(file_6_textlist)):
        attacked_password.append(file_6_textlist[i])
    for i in range(0, len(file_7_textlist)):
        attacked_password.append(file_7_textlist[i])
    
    file_1.close()
    file_2.close()
    file_3.close()
    file_4.close()
    file_5.close()
    file_6.close()
    file_7.close()
    
    label_txt = np.ones(len(attacked_password)) * -1
    
    all_data = np.concatenate((Password_data['password'], attacked_password), axis=0)
    all_label = np.concatenate((Password_data['strength'], label_txt), axis=None)
    print('all_data.shape: ', all_data.shape)
    
    all_data_ascii = np.zeros((18935625, max_len))
    print('all_data_ascii.shape: ', all_data_ascii.shape)
    for i in trange(0, len(all_data)):
        if len(str(all_data[i])) > max_len:
            all_data[i] = str(all_data[i])[0:max_len]
            for j in range(0, max_len):
                all_data_ascii[i][j] = ord(all_data[i][j])
        else:
            kalan_uzunluk = max_len - len(str(all_data[i]))
            bosluk = ''
            for j in range(0, kalan_uzunluk):
                bosluk = bosluk + ' '
            full_password = str(all_data[i]) +  bosluk
            all_data[i] = full_password
            for j in range(0, max_len):
                all_data_ascii[i][j] = ord(full_password[j])
    
    print('all_data.shape: ', all_data.shape)
    print('all_data_ascii.shape: ', all_data_ascii.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(all_data_ascii, all_label, test_size=0.2, random_state=42)
    
    print('all_data_ascii.shape: ', all_data_ascii.shape)
    
    print('Original dataset shape %s' % Counter(y_train))
    rus = RandomUnderSampler(random_state=42)
    X_res_rus, y_res_rus = rus.fit_resample(X_train, y_train)
    print('Resampled dataset shape %s' % Counter(y_res_rus))
    
    try:
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(X_res_rus, y_res_rus)
        print("neigh score: ", neigh.score(X_test, y_test))
        print(confusion_matrix(y_test, neigh.predict(X_test)))
        pickle.dump(neigh, open('Clf/knn.sav', 'wb'))
    except:
        print("error in knn")
    
    try:
        rf = RandomForestClassifier(random_state=0)
        rf.fit(X_res_rus, y_res_rus)
        print("RF score: ", rf.score(X_test, y_test))
        print(confusion_matrix(y_test, rf.predict(X_test)))
        pickle.dump(rf, open('Clf/rf.sav', 'wb'))
    except:
        print("error in rf")
   
prapera_data_to_text()
        
       






