#Comment
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from nltk.corpus import stopwords
from transformers import BertTokenizer,BertConfig
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy

import tensorflow
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
#tensorflow.autograph.set_verbosity(2)


from preprocessor import Preprocessor
from deep_models import DeepNLP
from config import Config
config = Config()


# This object contains the training and preprocessing pipelines in order to create a classification based on BERT Model.
class Classifier(DeepNLP):

    def __init__(self):
        
        DeepNLP.__init__(self)
        
        self.df = pd.DataFrame()
        self.X_train,self.X_test,self.y_train,self.y_test = None,None,None,None

        # Preprocessing params

        self.input_text_cols = [config.COLUMN_TITLE,config.COLUMN_DESCRIPTION,config.COLUMN_SOLUTION]
        self.y_col = config.COLUMN_IMPACT
        self.y_classes = None
        self.test_size = 0.2

        # Model params
        self.output_shape = None
        self.extrafeatures_shape = 2  

        # Compile params
        self.loss = 'categorical_crossentropy'
        self.metrics = CategoricalAccuracy('accuracy')

        #Callback params
        self.monitor = 'val_loss'
        self.patience = 1
        self.min_delta = 6e-1
        self.restore_best_weights = True

        # Train params
        self.batch_size = 32
        self.epochs = 100
        self.val_size = 0.1


        # Training loops params
        self.scaling = None
        self.l2_alpha = 0.005
        

        # Model
        self.model = None
        
    def training_pipe(self):
        
        model = self.deep_impact(training=True)
        model.summary()
        
        print(' # Model created')

        # Set an optimizer
        optimizer = Adam(
            learning_rate=5e-05,
            epsilon=1e-08,
            decay=0.01,
            clipnorm=1.0)

        # Compile the model
        model.compile(  loss = self.loss,
                        optimizer = optimizer, 
                        metrics = self.metrics)
        print(' # Model compiled')

        # Callback
        callback = EarlyStopping(monitor=self.monitor,
                                     min_delta=self.min_delta,
                                     patience=self.patience,
                                     restore_best_weights=self.restore_best_weights)

        # Train
        print("======= TRAIN MODEL =======")
        with tensorflow.device('/GPU:0'):
            print('Begining trainig...')
            history = model.fit(x=self.X_train,
                            y=self.y_train,
                            validation_split=self.val_size,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            callbacks=[callback])

        
        # Test
        print("======= TEST MODEL =======")
        # New model to deactivate dropout layers
        test_model = self.deep_impact(training=False)
        with tensorflow.device(config.device):
            test_model.set_weights(model.get_weights())
            test_model.compile(  loss = self.loss,
                        optimizer = optimizer, 
                        metrics = self.metrics)

        # Evaluation
        print("======= EVAL MODEL =======")
        with tensorflow.device(config.device):
            scores = test_model.evaluate(x=self.X_test, 
                                         y=self.y_test, 
                                         verbose=True)

        print(" # MODEL LOSS: {}\n # MODEL ACC: {}\n".format(scores[0],scores[1]))


        # ----- LOGGER ------

        # PLOT RESULTS
        # Section update under development in Plotter object.
        
        '''
        with tensorflow.device(config.device):
            y_test_pred = test_model.predict(self.X_test)
            y_train_pred = test_model.predict(self.X_train)
            
        fig1,ax1 = plt.subplots(1,1)
        self.plot_history(history,fig1,ax1,'loss')

        fig2,ax2 = plt.subplots(1,1)
        cfm = np.array(confusion_matrix(self.y_test.argmax(axis=1),self.y_test_pred.argmax(axis=1)))

        print(" # MODEL CONFUSION MATRIX: ")
        print(cfm)

        self.plot_confusion_matrix_custom(cm=cfm,fig=fig2,ax=ax2)

        plt.show()'''
 
    def preprocessing_pipe(self,df,add_extras=False,split=True):

    
        data = Preprocessor(df)
        
        # Select NLP Fields
        data.init(txt_cols=self.input_text_cols)

        # Output Transformation
        data.set_output(y_col=self.y_col,net_type='multiclass')

        self.Y_classes = data.y_classes
        self.output_shape = len(data.y_classes)

        if split == True:
            X_train,X_test,y_train,y_test = data.data_split(self.test_size)
        else:
            X_test,y_test = data.X,data.Y
       
        # EXTRA FEATURES NORMALIZATION
        if add_extras:
            
            df[config.COLUMN_SERVICE_CLASS] = df[config.COLUMN_SERVICE_CLASS].fillna(4)
            time_class = df[config.COLUMN_SERVICE_CLASS].values
            new_time_class = data.format_extra_feature(time_class,norm=True,norm_max=4)

            impact = df[config.COLUMN_IMPACT].values
            new_impact = data.format_extra_feature(impact,norm=True,norm_max=4)

            prio = df[config.COLUMN_PRIO].values
            new_prio = data.format_extra_feature(prio,norm=True,norm_max=8)
            
            hour = df[config.COLUMN_OPEN_TIME].dt.hour+df[config.COLUMN_OPEN_TIME].dt.minute/60
            new_hour = data.format_extra_feature(hour.values,norm=True,norm_max=24)

            month = df[config.COLUMN_OPEN_TIME].dt.month
            new_month = data.format_extra_feature(month.values,norm=True,norm_max=12)

            # NOTA: INCLUDE DAY OF THE WEEK!
            extra_features = np.concatenate((new_prio,
                                             new_impact,
                                             new_time_class,
                                             new_hour,
                                             new_month),axis=1)

            extra_features_train,extra_features_test = data.extra_data_split(extra_features,self.test_size)

            self.extrafeatures_shape = len(extra_features_test[0])

        # GET BETO MODEL VOCAB AND CONFIG
        # Load transformers config and set output_hidden_states to False
        model_config = BertConfig.from_json_file(config.MODEL_CONFIG)
        model_config.output_hidden_states = True
        
        # Load BERT tokenizer
        tokenizer_esp = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.MODEL_VOCAB,config=model_config)
        
        
        ### Format inputs data

        with tensorflow.device(config.device):
            
            print("### get_bert_inputs() start...")

            if split == True:
                encoded_data_test = data.get_bert_inputs(max_len = config.MAX_LEN,
                                                        tokenizer=tokenizer_esp,
                                                        data=X_test)
                encoded_data_train = data.get_bert_inputs(max_len = config.MAX_LEN,
                                                        tokenizer=tokenizer_esp,
                                                        data=X_train)
            else:
                encoded_data_test = data.get_bert_inputs(max_len = config.MAX_LEN,
                                                        tokenizer=tokenizer_esp,
                                                        data=X_test)

            print("### get_bert_inputs() end.")
            
            if split == True:

                X_test = {'input_ids': encoded_data_test[0],
                        'token_type_ids': encoded_data_test[1],
                        'attention_mask': encoded_data_test[2]}
                
                X_train = {'input_ids': encoded_data_train[0],
                        'token_type_ids': encoded_data_train[1],
                        'attention_mask': encoded_data_train[2]}

                if add_extras:
                    X_test['extra_features'] = extra_features_test
                    X_train['extra_features'] = extra_features_train
            else:
                
                X_test = {'input_ids': encoded_data_test[0],
                        'token_type_ids': encoded_data_test[1],
                        'attention_mask': encoded_data_test[2]}

                if add_extras:
                    X_test['extra_features'] = extra_features


        if split == True:
            self.X_train,self.X_test,self.y_train,self.y_test = X_train,X_test,y_train,y_test
        
        else:
            self.X_test,self.y_test = X_test,y_test


class Plotter():
    ''' Section under development
    '''
    def __init__(self):
        pass
    
    def plot_confusion_matrix_custom(self,cm,fig,ax,
                          target_names = ['1', '2', '3', '4'],
                          title = 'Confusion matrix',
                          cmap = None,
                          normalize = True):    
        

        #accuracy = np.trace(cm) / float(np.sum(cm))
        #misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')
        
        
        #plt.figure(figsize = (8, 6))
        ax.imshow(cm, interpolation = 'nearest', cmap = cmap)
        ax.set_title(title)
        #plt.colorbar()
        

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(target_names, rotation = 0)
            ax.set_yticks(tick_marks) 
            ax.set_yticklabels(target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                ax.text(j, i, "{:0.4f}".format(cm[i, j]),
                        horizontalalignment = "center",
                        color = "white" if cm[i, j] > thresh else "black")
            else:
                ax.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment = "center",
                        color = "white" if cm[i, j] > thresh else "black")

        
        #plt.tight_layout()
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        plt.grid(None)
          
    def plot_history(self,fit_history,figure,ax,metric):
      
        # summarize history for mse  
        train_r,val_r = ax.plot(fit_history.epoch,fit_history.history[metric],
                                fit_history.epoch,fit_history.history['val_'+metric])

        ax.legend((train_r,val_r),('train','val'),loc='upper right')

        ax.set_ylabel('Loss')
        ax.set_xlabel('Epochs')
        ax.set_title('Model Loss')#+' drop_out '+str(self.drop_out))
    
