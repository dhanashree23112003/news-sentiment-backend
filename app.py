from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load model from HuggingFace Hub
classifier = pipeline(
    "sentiment-analysis",
    model="dhanashree2311/news-distilroberta-sentiment"
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

import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
