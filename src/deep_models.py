

from config import Config
config = Config()

import tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate,AveragePooling1D,AveragePooling2D
from tensorflow.keras.initializers import TruncatedNormal
from transformers import BertConfig,TFBertModel

class DeepNLP():

    def __init__(self):

        self.output_shape = 2
        self.max_len = 256
        self.l2_alpha = 0.01
        self.drop_out = 0.2
        self.hidden_layer = 1
        self.output_layer = 1
        self.extrafeatures_shape = 1
        self.model_type = 'multiclass'
        
    def deep_impact(self,training=False):

        with tensorflow.device('/CPU:0'):
        
            # LOAD MODEL CONFIG
            model_config = BertConfig.from_json_file(config.MODEL_CONFIG)
            model_config.output_hidden_states = True

            # LOAD MODEL
            transformer_model = TFBertModel.from_pretrained(config.MODEL_PATH,config = model_config,from_pt=True)
            transformer_model.trainable = True # freeze BERT model weights
            bert = transformer_model.layers[0]
            
            
            # MODEL INPUTS
            #   BERT model inputs
            input_ids = Input(shape=(self.max_len), name='input_ids', dtype='int32')
            token_type_ids = Input(shape=(self.max_len), name='token_type_ids', dtype='int32')
            attention_mask = Input(shape=(self.max_len), name='attention_mask', dtype='int32')
            bert_inputs = {'input_ids': input_ids,
                            'token_type_ids':token_type_ids,
                            'attention_mask': attention_mask}
            
            #   Numerical data inputs
            extra_features = Input(shape=self.extrafeatures_shape,name='extra_features',dtype='float32')
            
            inputs= {'input_ids': input_ids,
                        'token_type_ids':token_type_ids,
                        'attention_mask': attention_mask,
                        'extra_features': extra_features}
            
            
            # MODEL ARCHITECTURE DEFINITION

            self.hidden_state = 2 # Get all BERT Layers outputs
            hidden_layers = bert(bert_inputs)[self.hidden_state]
            
            #   Mixing BERT intermediate layers
            hidden_layers = (hidden_layers[4],hidden_layers[5],hidden_layers[9],hidden_layers[10])
            hidden_layers = tensorflow.convert_to_tensor(list(hidden_layers))
            
            #   Layers permutation, required by syntax
            x = tensorflow.keras.backend.permute_dimensions(hidden_layers,(1,0,2,3)) 

            #   Sentence Embedding
            x = AveragePooling2D(pool_size=(self.max_len,1),data_format='channels_first',name='Sentence_Embedding')(x)
            x = tensorflow.squeeze(x,axis=2,name='Squeezer')

            #   Contatenating hidden layers sentence embeddings
            to_concat = []
            for i in range(x.shape[1]):
                to_concat.append(x[:,i,:])
            
            x = concatenate(to_concat,axis=-1)

            
            # CONCATENATING NLP AND EXTRAFEATURES BRANCHES

            x1 = extra_features
            x = concatenate([x,x1],axis=-1)
            
            x = Dense(3072,kernel_regularizer=regularizers.l2(self.l2_alpha),activation='relu')(x)
            x = Dropout(self.drop_out)(x,training=training)
            
            x = Dense(768,kernel_regularizer=regularizers.l2(self.l2_alpha),activation='relu')(x)
            x = Dropout(self.drop_out)(x,training=training)

            x = Dense(256,kernel_regularizer=regularizers.l2(self.l2_alpha),activation='relu',name='mbed256')(x) # Layer used for clustering if required
            x = Dropout(self.drop_out)(x,training=training)


            #   Model output selection depending on model type
            if self.model_type == 'regression':
                self.output_shape = 1
                outputs = Dense(units=self.output_shape,activation='relu',name='output_layer',kernel_initializer=TruncatedNormal(stddev=model_config.initializer_range))(x)

            elif self.model_type == 'multiclass':
                outputs = Dense(units=self.output_shape,activation='softmax', kernel_initializer=TruncatedNormal(stddev=model_config.initializer_range), name='output_layer')(x)
    
            #   And combine it all in a model object
            model = Model(inputs=inputs, outputs=outputs, name='nlp_regression')

            return model

    def model_dev(self,model_type,add_extras,training=False):

        with tensorflow.device('/CPU:0'):

            model_config = BertConfig.from_json_file(config.MODEL_CONFIG)
            model_config.output_hidden_states = True
            
            transformer_model = TFBertModel.from_pretrained(config.MODEL_PATH,config = model_config,from_pt=True)
            transformer_model.trainable = True
            
            
            bert = transformer_model.layers[0]
            
            
            # Build your model input
            input_ids = Input(shape=(self.max_len), name='input_ids', dtype='int32')
            token_type_ids = Input(shape=(self.max_len), name='token_type_ids', dtype='int32')
            attention_mask = Input(shape=(self.max_len), name='attention_mask', dtype='int32')

            bert_inputs = {'input_ids': input_ids,
                            'token_type_ids':token_type_ids,
                            'attention_mask': attention_mask}
            
            # Model Extrafeatures if required
            if  add_extras:
                extra_features = Input(shape=self.extrafeatures_shape,name='extra_features',dtype='float32')
                inputs= {'input_ids': input_ids,'token_type_ids':token_type_ids, 'attention_mask': attention_mask,'extra_features': extra_features}
            else:
                inputs = bert_inputs
            




            self.hidden_state = 2
            '''
                0 -> corresponds to BERT last layer
                1 -> corresponds to BERT last [CLS] layer
                2 -> corresponds to all BERT Layers
            '''

            #hidden_states = hidden_states[:,:,1:,:]

            if self.hidden_state != 1:
                hidden_layers = bert(bert_inputs)[self.hidden_state]
                
                # MIXING BERT INTERMEDIATE LAYERS

                hidden_layers = (hidden_layers[4],hidden_layers[5],hidden_layers[9],hidden_layers[10])
                hidden_layers = tensorflow.convert_to_tensor(list(hidden_layers)) # Uncomment if more than one output layer
               
                
                if hidden_layers.shape[0]!=None: # When studying multiple layers combinantions...
                    
                    

                    # Layers permutation, required by syntax
                    x = tensorflow.keras.backend.permute_dimensions(hidden_layers,(1,0,2,3))

                    # Sentence Embedding
                    x = AveragePooling2D(pool_size=(self.max_len,1),data_format='channels_first',name='Sentence_Embedding')(x)
                    x = tensorflow.squeeze(x,axis=2,name='Squeezer')

                    # Contatenating Hidden layers sentence embeddings
                    to_concat = []
                    for i in range(x.shape[1]):
                        to_concat.append(x[:,i,:])
                    
                    x = concatenate(to_concat,axis=-1)


                else: # When studying one single layer...
                    
                    x = AveragePooling1D(pool_size=(self.max_len),data_format='channels_last',name='Sentence_Embedding')(hidden_layers)
                    x = tensorflow.squeeze(x,axis=1,name='Squeezer')
            
            else:
                x = bert(bert_inputs)[1]
                pass

            
            
            # CONCATENATING NLP AND EXTRAFEATURES BRANCHES...

            if add_extras:
                x1 = extra_features

                x = concatenate([x,x1],axis=-1)
                
                x = Dense(3072,kernel_regularizer=regularizers.l2(self.l2_alpha),activation='relu')(x)
                x = Dropout(self.drop_out)(x,training=training)
                
                x = Dense(768,kernel_regularizer=regularizers.l2(self.l2_alpha),activation='relu')(x)
                x = Dropout(self.drop_out)(x,training=training)

                x = Dense(256,kernel_regularizer=regularizers.l2(self.l2_alpha),activation='relu',name='mbed256')(x) # Layer used for clustering if required
                x = Dropout(self.drop_out)(x,training=training)



            

            # Model output selection depending on model type
            if model_type == 'regression':
                self.output_shape = 1
                outputs = Dense(units=self.output_shape,activation='relu',name='output_layer',kernel_initializer=TruncatedNormal(stddev=model_config.initializer_range))(x)

            elif model_type == 'multiclass':
                outputs = Dense(units=self.output_shape,activation='softmax', kernel_initializer=TruncatedNormal(stddev=model_config.initializer_range), name='output_layer')(x)
                #outputs = Dense(units=self.output_shape,activation='softmax',kernel_regularizer=regularizers.l2(self.l2_alpha),name='output_layer')(x)
    
            # And combine it all in a model object
            model = Model(inputs=inputs, outputs=outputs, name='nlp_regression')

            return model



#deep_NLP = DeepNLP()
#deep_NLP.model('multiclass',True,True)