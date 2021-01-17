
from tqdm import tqdm


def get_sentence_embeddings(nlp, sentences):
    """ 
    Compute the embeddings of the sentences.
    """

    sentence_embeddings = []
    sentences_number = len(sentences)
    # Use tqdm to show progress bar
    with tqdm(total=sentences_number, leave=False) as progress_bar:
        for document in nlp.pipe(sentences, batch_size=50):
            # Get the embedding of the sentence
            sentence_embedding = document.vector
            sentence_embeddings.append(sentence_embedding)
            # Update the progress bar
            progress_bar.update(n=1)

    return sentence_embeddings
