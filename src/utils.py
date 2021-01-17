
from tqdm import tqdm


def get_embeddings_of_sentences(nlp, sentences):
    """ 
    Compute the embeddings of the sentences.
    """

    embeddings_of_sentences = []
    sentences_number = len(sentences)
    # Use tqdm to show progress bar
    with tqdm(total=sentences_number, leave=False) as pbar:
        for document in nlp.pipe(sentences, batch_size=50):
            # Get the embedding of the sentence
            sentence_embedding = document.vector
            embeddings_of_sentences.append(sentence_embedding)
            # Update the progress bar
            pbar.update(n=1)

    return embeddings_of_sentences
