
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import spacy_sentence_bert

from constants import DATASET_PATH, SPACY_MODEL_NAME, MODEL_PATH
from utils import generate_sentences_embeddings


def train_model():

    # Load dataset with the embeddings of the sentences in pandas DataFrame
    source_df = pd.read_csv(DATASET_PATH)[0:200]

    texts = list(source_df["text"])
    y = source_df["class"]

    # Split the dataset for training and test
    train_texts, test_texts, y_train, y_test = train_test_split(
                                                    texts, y, test_size=0.30, 
                                                    random_state=1)
    print(f'Number of sentences in the training datatet: {len(train_texts)}')
    
    nlp = spacy_sentence_bert.load_model(SPACY_MODEL_NAME)
    print('Getting embeddings of the sentences in the training datatet')
    X_train = generate_sentences_embeddings(nlp, train_texts)
    # Define the model
    model = LogisticRegression(random_state=0, solver='lbfgs',
                               multi_class='multinomial')

    # Train the model
    print('Training the model')
    model.fit(X_train, y_train)

    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved in {MODEL_PATH}')


if __name__ == "__main__":
    train_model()
