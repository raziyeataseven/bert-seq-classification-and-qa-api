import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from simpletransformers.classification import ClassificationModel


class TextClassificationModel:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        path =r'C:/Users/Administrator/graduate/bert_text_classification/assets'
        self.tokenizer  = AutoTokenizer.from_pretrained(path)

        self.model  = ClassificationModel('bert', r'C:/Users/Administrator/graduate/bert_text_classification/assets', use_cuda=False,  num_labels=7)
        self.classifier =  self.model

    def predict(self, text):
        with torch.no_grad():
            probabilities = self.classifier.predict([text]) #no need to apply softmax, bc predicted class is given at index obj[0][0]

        print ("prob object", probabilities) 
        print ("predicted class", probabilities[0][0]) 
        return probabilities[0][0]

# model = TextClassificationModel()

# def get_model():
#     return model