import tensorflow
import pandas as pd


from trainer import Classifier
from deep_models import DeepNLP


from config import Config
from config import cMSE
config = Config()

def main(net_type=None):

      
    if net_type == 'multiclass':

        tensorflow.config.set_soft_device_placement(True)
        
        df = pd.read_excel(config.TRAINING_DATA_PATH).sample(frac=0.01)
        
        ## Testing Model 
        bert = DeepNLP()
        bert.deep_impact(training=True)

        classifier = Classifier()

        # PREPROCESSING PIPELINE
        classifier.preprocessing_pipe(df=df,add_extras=True)
        print('### Preprocessing Pipeline DONE.')

        # TRAINING PIPELINE 
        classifier.training_pipe()
        print('### Training Pipeline DONE.')

        saved_model_name = config.SAVE_MODEL_TO + 'deep_impact_L-' + str(round(classifier.score[0],3)) +'_A-'+  str(round(classifier.score[1],3))      
        tensorflow.saved_model.save(classifier.model, export_dir=saved_model_name)
        print('## Model saved succesfully')
        
        

    else:
        print(" No model 'net_type' specified. Specify if 'multiclass'.")


main('multiclass')


