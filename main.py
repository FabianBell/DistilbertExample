from onnxruntime import InferenceSession
from transformers import DistilBertTokenizerFast
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset

def load_tokenizer():
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-german-cased')
    with open('additional_tokens', 'r') as fin:
        emojis = fin.read().split('\n')
        tokenizer.add_tokens(emojis)
    return tokenizer
        
def load_data():
    dataset = load_dataset("amazon_reviews_multi", "de", split="validation")
    data = dataset.to_pandas()
    data['label'] = None
    data.loc[data.stars > 3, 'label'] = 'positive'
    data.loc[data.stars == 3, 'label'] = 'neutral'
    data.loc[data.stars < 3, 'label'] = 'negative'
    data = data[['label', 'review_body']]
    data = data.rename({'review_body': 'input'}, axis=1)
    return data

def calculate_metric_for_label(label):
    ground_truth = data[data.label == label]
    predicted = data[data.prediction == label]
    correct = ground_truth[ground_truth.prediction == label]
    precision = len(correct) / (len(predicted) + 1e-10)
    recall = len(correct) / (len(ground_truth) + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    print('Metrics for label:', label)
    print('\tPrecision:', precision)
    print('\tRecall:', recall)
    print('\tF1:', f1)

tokenizer = load_tokenizer()
sentiment = ['positive', 'negative', 'neutral']
session = InferenceSession('models/distilbert.quant.onnx')
data = load_data()

predictions = []
for label, text in tqdm(data.iloc, total=len(data)):
    inp = dict(tokenizer(text, return_tensors='np'))
    prediction = session.run(None, inp)[0]
    predicted_sentiment = sentiment[prediction.argmax(-1)[0]]
    predictions.append(predicted_sentiment)

data['prediction'] = predictions
data.to_csv('output.csv')

# calculate metrics
accuracy = sum(data.label == data.prediction) / len(data)
print('Accuray:', accuracy)

# label wise metrics
for label in sentiment:
    calculate_metric_for_label(label)
