import json
from simpletransformers.classification import ClassificationModel
from torch import nn


class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        
        # importing classification model, using simpletransformers lib which installed at https://colab.research.google.com/drive
        #create model with turkish bert
        self.model = ClassificationModel('bert', r'C:/Users/Administrator/graduate/bert_text_classification/assets', use_cuda=False,  num_labels=7)
        
        print(self.model)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)

# if __name__=='__main__':
#     clf = BertClassifier(7)
#     print(clf.model.predict("kargo teslim edilemedi"))
