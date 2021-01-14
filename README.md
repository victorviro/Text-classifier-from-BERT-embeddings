
# Text classifier through the embeddings of a pretrained BERT model

In this repository, we train a text classifier through embeddings of sentences of a pretrained BERT model. To generate the embeddings of the sentences is recommended to use a GPU. Alternatively to this repository, there is a [jupyter notebook]() to reproduce this code in Google Colab with GPU-usage free.


Create a virtual environment via venv or conda e install requirements.
```
cd src
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Download the dataset.
```
wget https://bitbucket.org/delectateam/nlptrainingexam/raw/aa9ea86fa4795ef2bcba2af622add9a8e69c6621/resources/bert/corpus.csv -P data/raw
```


Train a simple logistic regression classifier. First, we get the embeddings of the sentences in the training dataset and then train the model. Finally, the model is saved in `models/`.
```
python src/train.py
```

Load and evaluate the model in the test dataset. First, we get the embeddings of the sentences in the test dataset and then evaluate the model computing some metrics. Finally, we show some predictions done by the model.
```
python src/evaluate.py
```
