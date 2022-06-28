
from typing import Any, Dict
from fastapi import FastAPI, Depends

from pydantic import BaseModel
from pydantic.schema import Optional
from .models.classification_model import TextClassificationModel
from .models.bert_qa import BertQA

app = FastAPI()


class ClassificationRequest(BaseModel):
    text: Optional[str]

class ClassificationResponse(BaseModel):
    predictedClass: Optional[int]
    originalText: Optional[str]

class QuestionAnsweringRequest(BaseModel):
    question: Optional[str]
    answer_content: Optional[str]

class QuestionAnsweringResponse(BaseModel):
    answer: Optional[str]


@app.post("/predict", response_model=ClassificationResponse)
def predict(request: ClassificationRequest):
    model = TextClassificationModel()    
    predictedClass = model.predict(request.text)
    return ClassificationResponse(predictedClass=predictedClass, originalText=request.text)
    

@app.post("/answer", response_model=QuestionAnsweringResponse)
def predict(request: QuestionAnsweringRequest):
    model = BertQA()
    answer =  model.answer_question(request.question, request.answer_content)
    return QuestionAnsweringResponse(answer=answer, originalQuestion=request.question)

@app.get('/test', response_model=str)
def test():
    print ("I am at the endpoint, test", )
    return "text 123"
