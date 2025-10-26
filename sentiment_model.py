from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import streamlit as st

@st.cache_resource
def load_model():
    # Always allocate tensors on a real CPU device, not meta
    torch.set_default_device("cpu")

    model_name = "ProsusAI/finbert"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float32,      # disable half precision
        low_cpu_mem_usage=False,        # ensures proper CPU allocation
        device_map=None                 # disables auto-offload to meta
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
    return nlp


def analyze_sentiment(df):
    sentiment_pipeline = load_model()
    sentiments = []
    for text in df["title"]:
        if isinstance(text, str):
            try:
                res = sentiment_pipeline(text[:512])[0]
                sentiments.append(res["label"])
            except Exception:
                sentiments.append("NEUTRAL")
        else:
            sentiments.append("NEUTRAL")
    df["sentiment"] = sentiments
    return df, df["sentiment"].value_counts().to_dict()
