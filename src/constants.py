import os


PROJECT_PATH = os.getcwd()
DATASET_PATH = f'{PROJECT_PATH}/data/raw/corpus.csv'

SENTENCES_EMBEDDINGS_PATH = (f'{PROJECT_PATH}/data/processed/'
                             'sentences_embeddings.csv')

SPACY_MODEL_NAME = 'en_bert_base_nli_cls_token'

MODEL_PATH = f'{PROJECT_PATH}/models/model.pkl'

# Number of predictions to show when evaluate the model
N_PREDICTIONS_TO_SHOW = 10
