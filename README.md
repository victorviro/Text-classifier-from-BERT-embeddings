
# Text classifier through the embeddings of a pre-trained BERT model

## Description

In this repository, we train a text binary classifier through sentence embeddings of a pre-trained **BERT** model. 

## Details

The dataset consists of 60000 paragraphs labeled in two categories: world and sports. The dataset is balanced, with approximately the same number of examples for each label.

We get the embeddings of the sentences from the [sentence-transformers](https://github.com/UKPLab/sentence-transformers) package directly from *spaCy* using [spacy_sentence_bert](https://spacy.io/universe/project/spacy-sentence-bert) (a review of text vectorization techniques, including contextualized word/sentence embeddings, is available [here](https://nbviewer.jupyter.org/github/victorviro/Deep_learning_python/blob/master/Text_Vectorization_NLP.ipynb)). To generate the embeddings of the sentences fast is recommended to use a GPU. Alternatively to this repository, a [jupyter notebook](https://colab.research.google.com/drive/1gafjfsBsmZYiTNG8V4OXiwiQNh7FdhsT?usp=sharing) is available in Google Colab with GPU-usage free to reproduce this code with additional exploratory data analysis.

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

The model gets an accuracy of `0.97` in the test dataset. The next figure shows the confusion matrix in the test dataset.


![](https://i.ibb.co/8XYZx5r/error-matrix.png)
