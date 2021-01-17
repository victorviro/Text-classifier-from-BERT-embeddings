
# Text classifier through the embeddings of a pretrained BERT model

## Description

In this repository, we train a text binary classifier through sentence embeddings of a pretrained **BERT** model. 

## Details

We get the embeddings from the [sentence-transformers](https://github.com/UKPLab/sentence-transformers) package directly from *spaCy* using [spacy_sentence_bert](https://spacy.io/universe/project/spacy-sentence-bert) (a review of contextualized word/sentence embeddings is available [here](https://colab.research.google.com/drive/13wnd2aB7w2NCdMp-mE1k9XwsqbHjhbNM?usp=sharing)). To generate the embeddings of the sentences fast is recommended to use a GPU. Alternatively to this repository, a [jupyter notebook](https://colab.research.google.com/drive/1gafjfsBsmZYiTNG8V4OXiwiQNh7FdhsT?usp=sharing) is available to reproduce this code in Google Colab with GPU-usage free.

## Set up
Download the dataset.
```shell
wget https://bitbucket.org/delectateam/nlptrainingexam/raw/aa9ea86fa4795ef2bcba2af622add9a8e69c6621/resources/bert/corpus.csv -P data/raw
```

Create a virtual environment, and install requirements.

```shell
cd src
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Training
Train a logistic regression classifier. 
- First, we get the embeddings of the sentences in the training dataset.
- Then, train the model (it will be saved in `models/`).

```shell
python src/train.py
```

## Evaluation
Load and evaluate the model in the test dataset. 
- First, we get the embeddings of the sentences in the test dataset. 
- Then, evaluate the model in the test dataset computing some metrics. 
- Finally, we show some predictions done by the model.

```shell
python src/evaluate.py
```
