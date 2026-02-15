from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load model from HuggingFace Hub
classifier = pipeline(
    "sentiment-analysis",
    model="dhanashree2311/news-sentimnet-roberta"
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "News Sentiment API Running"}

@app.post("/predict")
def predict(request: TextRequest):
    result = classifier(request.text)[0]

    label = result["label"]
    score = result["score"]

    return {
        "label": label,
        "score": score
    }
