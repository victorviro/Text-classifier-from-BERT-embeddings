
from tqdm import tqdm


def generate_sentences_embeddings(nlp, texts):
    """ 
    Compute the embeddings of the sentences.
    """

    embeddings_of_sentences = []
    texts_number = len(texts)
    with tqdm(total=texts_number, leave=False) as pbar:
        for doc in nlp.pipe(texts, batch_size=50):
            # Get the embedding of the sentence
            text_embedding = doc.vector
            embeddings_of_sentences.append(text_embedding)
            pbar.update(n=1)
    print('Obtained embeddings of the sentences')

    return embeddings_of_sentences
