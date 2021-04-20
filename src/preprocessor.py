

import numpy as np
import pandas as pd
import string

from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer # Tokenizer to convert words into numbers
from tensorflow.keras.preprocessing.text import text_to_word_sequence # Converts string to word sequence
from sklearn.preprocessing import LabelEncoder #For one-hot encoder
from tensorflow.keras.utils import to_categorical # One-hot encoding
from sklearn.model_selection import train_test_split # Separes test and train dataset
from sklearn.preprocessing import MinMaxScaler

#from transformers import BertForSequenceClassification, BertTokenizer

# --- Neural network libraries

from tensorflow.keras.preprocessing.sequence import pad_sequences # Normlize array for squared NN input

from config import Config

config = Config()



class Preprocessor():

    def __init__(self,data):
        
        self.df = data # input dataframe
        self.Y = -1 # output
        self.X = -1 # input 

        self.X_train, self.X_test, self.y_train, self.y_test = (-1,-1,-1,-1)


        self.words = -1 # dictionary of which contains the vocabulary
        self.word_counts = -1 # most repeated words
        self.y_classes = None
        self.scaling = MinMaxScaler(feature_range=(0,12))      

    def init(self,txt_cols=None):


        self.df = self.df.reset_index(drop=True)
        self.df['all_text'] = ''
        self.df[config.COLUMN_TITLE]= self.df[config.COLUMN_TITLE].fillna('').astype(str)
        self.df[config.COLUMN_DESCRIPTION]= self.df[config.COLUMN_DESCRIPTION].fillna('').astype(str)
        self.df[config.COLUMN_SOLUTION]= self.df[config.COLUMN_SOLUTION].fillna('').astype(str)

        for count, row in self.df.iterrows():
            for col in txt_cols:
                self.df['all_text'].iloc[count] =  self.df['all_text'].iloc[count] +  self.remove_punctuation(self.df[col].iloc[count])+' '
        
        self.X = self.df['all_text'].str.lower()
        
        sequences = []
        
        i = 0
        for seq in self.X:
            ci_name = self.df[config.COLUMN_CI].str.lower().iloc[i]
            text = np.append(ci_name,seq.split(' '))
            sequences.append(text)
            i+=1

        self.X = sequences

    def set_output(self,y_col,net_type):

        if net_type == 'multiclass':
            self.Y = self.df[y_col]
            # encode class values as integers
            encoder = LabelEncoder()
            encoder.fit(self.Y)
            self.Y = encoder.transform(self.Y)
            # convert integers to dummy variables (i.e. one hot encoded)
            self.Y = to_categorical(self.Y)
            self.y_classes = encoder.classes_
            print('## OUTPUT LABELS: ', self.y_classes)

        elif net_type == 'regression':

            out_linear = (self.df[y_col]/60).astype(int)
            to_scale = np.reshape(out_linear.values,(len(out_linear),1))
            self.Y = self.scaling.fit_transform(to_scale)
            self.Y = np.reshape(self.Y,(len(self.Y))) 

    def remove_punctuation(self,txt):

        ''' This functions removes punctuation from text
        '''
        # remove punctuation
        non_words = list(string.punctuation)
        #we add spanish punctuation
        non_words.extend(['¿', '¡'])
        non_words.extend(map(str,range(10)))
        no_punct = ''.join([c for c in txt.replace('\n',' ' ) if c not in non_words])
        
        return no_punct

    def remove_stopwords(self,stop_words,new_stop_words):

        ''' This functions removes new_stop_words to the predefined language stopwords
        '''

        print("Function executed: remove_stopwords()")

        self.stop_words.extend(stop_words)
        self.stop_words.extend(new_stop_words)

        array = self.X_seq
        stoped_array = []
        for txt in array: 
            words= [w for w in txt if w not in self.stop_words]
            stoped_array.append(words)
        
        c_empty = 0
        for i in range(len(stoped_array)):
            txt = stoped_array[i]
            if len(txt) == 0:
                stoped_array[i] = 'empty_text'
                c_empty +=1
            self.X_lens.append(len(txt))
                
        self.X = stoped_array
        self.X_seq = stoped_array

        # in order to put new stop_words
        tokenizer = Tokenizer(num_words=2000)
        tokenizer.fit_on_texts(stoped_array)
        self.word_counts = dict(tokenizer.word_counts)
        self.word_counts = {key: value for key, value in reversed(sorted(self.word_counts.items(), key=lambda item: item[1]))}

        #word_counts_df = pd.DataFrame.from_dict(self.word_counts)
        #word_counts_df.to_excel('word_counts_df.xlsx')
    
    def format_extra_feature(self,feature,labels=None,norm=False,fit=False,norm_max=0,ohe=False,n_classes=None):

        if norm_max==0 and norm==True:
            print("In format_extra_features fit was set to True whiel no fit_max was given")
            return -1
            
        if fit == True:
            encoder = LabelEncoder()
            encoder.fit(labels)
            feature = encoder.transform(feature)

        if ohe == True: 
            feature = np.array(to_categorical(feature,n_classes))
        
        if norm == True: 
            myInt = norm_max
            new_feature = [float(x) / float(myInt) for x in feature]
            feature = np.reshape(new_feature,(len(new_feature),1))

        return feature

    def get_bert_inputs(self,max_len,tokenizer,data):
        
        input_ids_list = np.empty((0,config.MAX_LEN))
        token_type_ids_list = np.empty((0,config.MAX_LEN))
        attention_masks_list = np.empty((0,config.MAX_LEN))
    
        for seq in data:


            encoded_dic = tokenizer.encode_plus(
                text=list(seq),
                add_special_tokens=True,
                max_length=max_len,
                #padding=True, 
                pad_to_max_length=True,
                return_tensors='tf',
                return_token_type_ids = True,
                return_attention_mask = True,
                )#verbose=True)
    

            
            input_ids_list = np.append(input_ids_list,np.array(encoded_dic['input_ids']),axis=0)
            token_type_ids_list = np.append(token_type_ids_list,np.array(encoded_dic['token_type_ids']),axis=0)
            attention_masks_list = np.append(attention_masks_list,np.array(encoded_dic['attention_mask']),axis=0)
                
        
        input_ids_list = np.reshape(input_ids_list,(len(input_ids_list),max_len))
        token_type_ids_list = np.reshape(token_type_ids_list,(len(token_type_ids_list),max_len))
        attention_masks_list = np.reshape(attention_masks_list,(len(attention_masks_list),max_len))

        return [input_ids_list,token_type_ids_list,attention_masks_list]

    def extra_data_split(self,datain,test_size):

        X_train, X_test, _, _ = train_test_split(datain,datain,test_size=test_size, random_state=42)

        return X_train, X_test
 
    def data_split(self,test_size):

        print("Function executed: data_split()")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.Y, 
                                                                                test_size=test_size,
                                                                                random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test

