import pickle
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, classification_report, 
                             confusion_matrix)
import spacy_sentence_bert

from constants import (DATASET_PATH, SPACY_MODEL_NAME, MODEL_PATH,
                       N_PREDICTIONS_TO_SHOW)
from utils import get_embeddings_of_sentences


def evaluate_model():
    """
    Load the dataset, get the embeddings for the sentences in the 
    test dataset, load the model trained and compute evaluation 
    metrics in the test dataset. Finally, show some predictions done 
    by the model in the test dataset.
    """ 

    # Load dataset in pandas DataFrame
    source_df = pd.read_csv(DATASET_PATH)

    # Get the sentences and the target variable separately
    texts = list(source_df["text"])
    y = source_df["class"]

    # Split the dataset for training and test
    train_texts, test_texts, y_train, y_test = train_test_split(
                                                    texts, y, test_size=0.30, 
                                                    random_state=1)
    print(f'Number of sentences in the test datatet: {len(test_texts)}')

    # Load the spaCy statistical model 
    print('Loading the spaCy statistical model...')
    nlp = spacy_sentence_bert.load_model(SPACY_MODEL_NAME)

    print('Getting embeddings of the sentences in the test datatet')
    X_test = get_embeddings_of_sentences(nlp, test_texts)
    print('Obtained embeddings of the sentences')

    # Load the model trained
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print('Model loaded')

    # Get predictions in the test dataset
    y_test_predictions = model.predict(X_test)
    
    # Compute metrics to evaluate the model
    classification_metrics = classification_report(y_test, y_test_predictions)
    # Compute the area under the ROC curve
    y_test_probabilities = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_test_probabilities[:,1:2], multi_class="ovr")
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_predictions, labels=[2,1])
    print('\nMetrics:')
    print(f'Area under the ROC curve: {roc_auc}')
    print(f'Classification metrics:\n{classification_metrics}')
    print(f'Confusion matrix:\n{conf_matrix}')

    # Show some predictions in the test dataset
    print(f'\nShow {N_PREDICTIONS_TO_SHOW} predictions in the test dataset')

    for text, ground_truth in random.sample(list(zip(test_texts, y_test.values)), 
                                            k=N_PREDICTIONS_TO_SHOW):
        # Get the embedding of the sentence
        sentence_embedding = nlp(text).vector
        # Get the value predicted by the model
        prediction = model.predict([sentence_embedding])[0]

        print(f'\nText: {text}')
        print(f'Ground truth: {"Sports" if ground_truth==2 else "World"}')
        print(f'Predicted: {"Sports" if prediction==2 else "World"}')

if __name__ == "__main__":
    evaluate_model()
