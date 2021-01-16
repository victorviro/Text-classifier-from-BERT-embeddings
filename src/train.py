
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import spacy_sentence_bert

from constants import DATASET_PATH, SPACY_MODEL_NAME, MODEL_PATH
from utils import get_embeddings_of_sentences


def train_model():
    """
    Load the dataset, get the embeddings for the sentences in the 
    training dataset, train the model and save it.
    """ 

    # Load dataset in pandas DataFrame
    source_df = pd.read_csv(DATASET_PATH)

    # Get the sentences and the target variable separately
    texts = list(source_df["text"])
    y = source_df["class"]

    # Split the dataset for training and test
    train_texts, test_texts, y_train, y_test = train_test_split(
                                                    texts, y, test_size=0.3, 
                                                    random_state=1)
    print(f'Number of sentences in the train datatet: {len(train_texts)}')

    # Load the spaCy statistical model 
    print('Loading the spaCy statistical model...')
    nlp = spacy_sentence_bert.load_model(SPACY_MODEL_NAME)

    print('Getting embeddings of the sentences in the training datatet')
    X_train = get_embeddings_of_sentences(nlp, train_texts)
    print('Obtained embeddings of the sentences')

    # Define the model
    model = LogisticRegression(random_state=0, max_iter=1000)

    # Train the model
    print('Training the model')
    model.fit(X_train, y_train)

    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved in {MODEL_PATH}')


if __name__ == "__main__":
    train_model()
