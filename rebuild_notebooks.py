import json
from pathlib import Path

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(True)}

def write_nb(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.x"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(nb, indent=2))

sentiment_cells = [
    md("""
# Aspect-Aware Sentiment Modeling for Educational Feedback
Formalizes the research-exchange idea with runnable baselines, advanced models, prompting hooks, and explainability for joint aspect+sentiment.
"""),
    md("""
## 1. Setup
This notebook provides executable code blocks from data loading through training, evaluation, explainability, and CLI entry points. Install extras (`transformers`, `sentence-transformers`, `shap`, `lime`, `openai`) if they are not already available in your environment.
"""),
    code("""
import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

import shap
from lime.lime_text import LimeTextExplainer

# Optional heavy deps (guarded imports)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )
    import datasets
    from torch import nn
    import torch
except Exception:
    AutoTokenizer = AutoModelForSequenceClassification = Trainer = TrainingArguments = DataCollatorWithPadding = None
    datasets = nn = torch = None
"""),
    code("""
@dataclass
class Config:
    data_path: Path = Path("data_feedback.xlsx")
    text_col: str = "comments"
    aspect_col: str = "teacher/course"
    label_col: str = "sentiment"
    random_state: int = 42
"""),
    code("""
def load_data(cfg: Config) -> pd.DataFrame:
    if cfg.data_path.exists():
        df = pd.read_excel(cfg.data_path)
    else:
        df = pd.DataFrame(
            {
                "teacher/course": ["teacher", "course"],
                "comments": ["great teacher", "great course"],
                "sentiment": ["positive", "positive"],
            }
        )
    df = df.rename(columns={cfg.text_col: "text", cfg.aspect_col: "aspect_tag", cfg.label_col: "label"})
    df = df.dropna(subset=["text", "aspect_tag", "label"]).reset_index(drop=True)
    return df
"""),
    md("""
## 2. Data audit and splits
Includes aspect-specific splits to enable cross-aspect transfer evaluation.
"""),
    code("""
def train_val_split(df: pd.DataFrame, cfg: Config):
    strat = df["label"] if df["label"].nunique() > 1 else None
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=cfg.random_state, stratify=strat)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

cfg = Config()
df = load_data(cfg)
train_df, val_df = train_val_split(df, cfg)
train_df.head(), val_df.head()
"""),
    md("""
## 3. Preprocessing helpers and evaluation utilities
"""),
    code("""
def prepend_aspect(texts: List[str], aspects: List[str]):
    return [f"[ASPECT={a}] {t}" for t, a in zip(texts, aspects)]


def evaluate(model, X_val, y_val, target_names=None, label="eval"):
    preds = model.predict(X_val)
    report = classification_report(y_val, preds, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_val, preds)
    print(f"\n==== {label} report ====")
    print(report)
    print("Confusion matrix:\n", cm)
    return report, cm
"""),
    md("""
## 4. N-gram TF–IDF baselines (word + character)
Includes aspect prompts to test aspect-aware conditioning.
"""),
    code("""
def run_tfidf_baseline(train_df, val_df, use_aspect_prompt=False, analyzer="word", ngram_range=(1, 2)):
    X_train = train_df["text"] if not use_aspect_prompt else prepend_aspect(train_df["text"].tolist(), train_df["aspect_tag"].tolist())
    X_val = val_df["text"] if not use_aspect_prompt else prepend_aspect(val_df["text"].tolist(), val_df["aspect_tag"].tolist())
    y_train, y_val = train_df["label"], val_df["label"]

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=1)),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe, evaluate(pipe, X_val, y_val, label=f"tfidf-{analyzer}-aspect={use_aspect_prompt}")
"""),
    md("""
## 5. Sentence-embedding classifier (SBERT + LogisticRegression)
Uses aspect prompts to provide aspect-aware context.
"""),
    code("""
def run_sbert_classifier(train_df, val_df, model_name: str = "all-MiniLM-L6-v2", use_aspect_prompt=True):
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not installed")

    model = SentenceTransformer(model_name)
    X_train = train_df["text"] if not use_aspect_prompt else prepend_aspect(train_df["text"].tolist(), train_df["aspect_tag"].tolist())
    X_val = val_df["text"] if not use_aspect_prompt else prepend_aspect(val_df["text"].tolist(), val_df["aspect_tag"].tolist())
    y_train, y_val = train_df["label"], val_df["label"]

    train_emb = model.encode(list(X_train), batch_size=16, show_progress_bar=True)
    val_emb = model.encode(list(X_val), batch_size=16, show_progress_bar=True)

    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(train_emb, y_train)
    evaluate(clf, val_emb, y_val, label=f"sbert-aspect={use_aspect_prompt}")
    return clf, model
"""),
    md("""
## 6. Transformer fine-tuning (aspect-prompted)
Lightweight Trainer setup for reproducibility.
"""),
    code("""
def prepare_hf_dataset(train_df, val_df, tokenizer):
    def tokenize(batch):
        texts = prepend_aspect(batch["text"], batch["aspect_tag"])
        return tokenizer(texts, truncation=True, max_length=256)

    train_ds = datasets.Dataset.from_pandas(train_df)
    val_ds = datasets.Dataset.from_pandas(val_df)
    return train_ds.map(tokenize, batched=True), val_ds.map(tokenize, batched=True)


def run_transformer(train_df, val_df, model_name="distilbert-base-uncased", num_epochs=3):
    if AutoTokenizer is None:
        raise ImportError("transformers not installed")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=train_df["label"].nunique())

    train_ds, val_ds = prepare_hf_dataset(train_df, val_df, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir="./outputs",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=50,
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        pred_labels = preds.argmax(axis=1)
        report = classification_report(labels, pred_labels, output_dict=True, zero_division=0)
        return {"macro_f1": report["macro avg"]["f1-score"]}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer
"""),
    md("""
## 7. Multi-task (aspect + sentiment) head for joint learning
Shared encoder with dual classification heads to exploit aspect cues when predicting sentiment.
"""),
    code("""
class MultiTaskHead(nn.Module):
    def __init__(self, hidden_size, num_sentiment, num_aspect):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiment)
        self.aspect_classifier = nn.Linear(hidden_size, num_aspect)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        return {
            "sentiment": self.sentiment_classifier(x),
            "aspect": self.aspect_classifier(x),
        }
"""),
    md("""
## 8. Cross-aspect robustness (train on teacher, test on course and vice versa)
"""),
    code("""
def cross_aspect_eval(df, model_fn):
    teacher_df = df[df["aspect_tag"] == "teacher"].reset_index(drop=True)
    course_df = df[df["aspect_tag"] == "course"].reset_index(drop=True)
    results = {}
    for train_df, test_df, tag in [
        (teacher_df, course_df, "train_teacher_test_course"),
        (course_df, teacher_df, "train_course_test_teacher"),
    ]:
        model, _ = model_fn(train_df, test_df)
        results[tag] = model
    return results
"""),
    md("""
## 9. Lightweight data augmentation for robustness
Simple synonym/word-drop augmentations; plug into any experiment.
"""),
    code("""
def random_drop(text, p=0.15):
    words = text.split()
    keep = [w for w in words if random.random() > p]
    return " ".join(keep) if keep else text


def augment_dataframe(df, times=1):
    rows = []
    for _ in range(times):
        for _, row in df.iterrows():
            rows.append({
                "text": random_drop(row["text"]),
                "aspect_tag": row["aspect_tag"],
                "label": row["label"],
            })
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
"""),
    md("""
## 10. Zero/low-shot prompting baseline (LLM)
Uses explicit schema and aspect cues; keep API keys in environment variables.
"""),
    code("""
def prompt_sentiment(texts: List[str], aspects: List[str], model_name: str = "gpt-4o-mini"):
    import openai

    client = openai.OpenAI()
    outputs = []
    for t, a in zip(texts, aspects):
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Classify sentiment as positive, neutral, or negative and explain briefly."},
                {"role": "user", "content": f"Aspect: {a}. Comment: {t}"},
            ],
            temperature=0,
        )
        outputs.append(res.choices[0].message["content"])
    return outputs
"""),
    md("""
## 11. Error analysis and explainability reports
Combine SHAP/LIME outputs with per-length/per-aspect slices.
"""),
    code("""
def error_table(model, val_df, use_aspect_prompt=True):
    X_val = val_df["text"] if not use_aspect_prompt else prepend_aspect(val_df["text"].tolist(), val_df["aspect_tag"].tolist())
    y_true = val_df["label"].tolist()
    preds = model.predict(X_val)
    df_err = val_df.copy()
    df_err["pred"] = preds
    df_err["correct"] = df_err["pred"] == df_err["label"]
    return df_err.sort_values("correct")


def explain_with_shap(model, X_samples: List[str], class_names: List[str]):
    explainer = shap.Explainer(model.predict_proba, masker=shap.maskers.Text())
    shap_values = explainer(X_samples)
    shap.plots.text(shap_values, display=False)
    return shap_values


def explain_with_lime(model, X_samples: List[str], class_names: List[str]):
    explainer = LimeTextExplainer(class_names=class_names)
    explanations = [explainer.explain_instance(x, model.predict_proba, num_features=8) for x in X_samples]
    return explanations
"""),
    md("""
## 12. CLI entry points
Run from terminal: `python -m sentiment_analysis --model tfidf` etc.
"""),
    code("""
def main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Aspect-aware sentiment experiments")
    parser.add_argument("--model", choices=["tfidf", "char", "sbert", "transformer"], default="tfidf")
    args = parser.parse_args()

    cfg = Config()
    df = load_data(cfg)
    train_df, val_df = train_val_split(df, cfg)

    if args.model == "tfidf":
        run_tfidf_baseline(train_df, val_df, use_aspect_prompt=False)
    elif args.model == "char":
        run_tfidf_baseline(train_df, val_df, use_aspect_prompt=True, analyzer="char", ngram_range=(3, 5))
    elif args.model == "sbert":
        run_sbert_classifier(train_df, val_df)
    elif args.model == "transformer":
        run_transformer(train_df, val_df)


if __name__ == "__main__":
    main_cli()
"""),
]

write_nb("sentiment_analysis.ipynb", sentiment_cells)

aspect_cells = [
    md("""
# Aspect Classification for Educational Feedback
Aspect-focused notebook with baselines, prompting, cross-domain checks, and explainability.
"""),
    md("""
## 1. Setup
Load data, define helper utilities, and prepare aspect glossary for prompts and interpretability.
"""),
    code("""
import json
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import shap
from lime.lime_text import LimeTextExplainer

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
"""),
    code("""
def load_aspect_data(path=Path("data_feedback.xlsx")):
    if path.exists():
        df = pd.read_excel(path)
    else:
        df = pd.DataFrame(
            {
                "teacher/course": ["teacher", "course"],
                "comments": ["great teacher", "great course"],
                "aspect": ["general", "relevancy"],
            }
        )
    df = df.rename(columns={"comments": "text", "teacher/course": "topic"})
    return df.dropna(subset=["text", "aspect"])
"""),
    md("""
## 2. Glossary and preprocessing helpers
"""),
    code("""
aspect_glossary = {
    "teaching skills": "Pedagogical clarity, examples, pacing, interaction.",
    "behaviour": "Politeness, respect, supportiveness, responsiveness.",
    "knowledge": "Depth and breadth of subject knowledge.",
    "relevancy": "Alignment of content and practice with course goals.",
    "general": "General praise or criticism without a specific aspect.",
}


def prepend_topic(texts: List[str], topics: List[str]):
    return [f"[TOPIC={t}] {x}" for x, t in zip(texts, topics)]
"""),
    md("""
## 3. TF–IDF baselines (word + char) with explainability
"""),
    code("""
def run_aspect_tfidf(train_df, val_df, analyzer="word", prompt_topic=False):
    X_train = train_df["text"] if not prompt_topic else prepend_topic(train_df["text"].tolist(), train_df["topic"].tolist())
    X_val = val_df["text"] if not prompt_topic else prepend_topic(val_df["text"].tolist(), val_df["topic"].tolist())
    y_train, y_val = train_df["aspect"], val_df["aspect"]

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    print(classification_report(y_val, preds, zero_division=0))
    print(confusion_matrix(y_val, preds))

    explainer = LimeTextExplainer(class_names=pipe.classes_)
    explanation = explainer.explain_instance(X_val.iloc[0], pipe.predict_proba, num_features=8)
    return pipe, explanation
"""),
    md("""
## 4. Sentence embedding baseline (SBERT)
"""),
    code("""
def run_aspect_sbert(train_df, val_df, model_name="all-MiniLM-L6-v2", prompt_topic=True):
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not installed")

    model = SentenceTransformer(model_name)
    X_train = train_df["text"] if not prompt_topic else prepend_topic(train_df["text"].tolist(), train_df["topic"].tolist())
    X_val = val_df["text"] if not prompt_topic else prepend_topic(val_df["text"].tolist(), val_df["topic"].tolist())
    y_train, y_val = train_df["aspect"], val_df["aspect"]

    train_emb = model.encode(list(X_train), batch_size=16, show_progress_bar=True)
    val_emb = model.encode(list(X_val), batch_size=16, show_progress_bar=True)

    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(train_emb, y_train)
    preds = clf.predict(val_emb)
    print(classification_report(y_val, preds, zero_division=0))
    return clf, model
"""),
    md("""
## 5. Cross-domain / cross-topic evaluation
Train on teacher-only vs course-only to measure transfer.
"""),
    code("""
def cross_domain(train_df, test_df, model_fn):
    model = model_fn(train_df, test_df)
    return model
"""),
    md("""
## 6. Prompting baseline
Zero/low-shot LLM baseline using aspect definitions.
"""),
    code("""
def prompt_aspect(texts: List[str], model_name="gpt-4o-mini"):
    import openai

    system_msg = "Identify the aspect label using the glossary and respond with the label only."
    client = openai.OpenAI()
    outputs = []
    for t in texts:
        res = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": t}],
            temperature=0,
        )
        outputs.append(res.choices[0].message["content"])
    return outputs
"""),
    md("""
## 7. Explainability utilities
"""),
    code("""
def shap_for_aspect(model, X_samples: List[str]):
    explainer = shap.Explainer(model.predict_proba, masker=shap.maskers.Text())
    values = explainer(X_samples)
    shap.plots.text(values, display=False)
    return values
"""),
    md("""
## 8. Error inspection helpers
"""),
    code("""
def error_breakdown(model, val_df, prompt_topic=False):
    X_val = val_df["text"] if not prompt_topic else prepend_topic(val_df["text"].tolist(), val_df["topic"].tolist())
    preds = model.predict(X_val)
    df_err = val_df.copy()
    df_err["pred"] = preds
    df_err["correct"] = df_err["pred"] == df_err["aspect"]
    return df_err.sort_values("correct")
"""),
    md("""
## 9. CLI entry point
"""),
    code("""
def main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Aspect classification experiments")
    parser.add_argument("--model", choices=["tfidf", "char", "sbert"], default="tfidf")
    args = parser.parse_args()

    df = load_aspect_data()
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["aspect"])

    if args.model == "tfidf":
        run_aspect_tfidf(train_df, val_df, analyzer="word", prompt_topic=True)
    elif args.model == "char":
        run_aspect_tfidf(train_df, val_df, analyzer="char", prompt_topic=True)
    elif args.model == "sbert":
        run_aspect_sbert(train_df, val_df)


if __name__ == "__main__":
    main_cli()
"""),
]

write_nb("aspect_classification.ipynb", aspect_cells)

teacher_course_cells = [
    md("""
# Teacher vs Course Classification
Notebook for topic discrimination with explainable baselines, prompting, and robustness slices.
"""),
    md("""
## 1. Setup
"""),
    code("""
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import shap
from lime.lime_text import LimeTextExplainer

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
"""),
    md("""
## 2. Load data
"""),
    code("""
def load_topic_data(path=Path("data_feedback.xlsx")):
    if path.exists():
        df = pd.read_excel(path)
    else:
        df = pd.DataFrame(
            {
                "teacher/course": ["teacher", "course"],
                "comments": ["great teacher", "great course"],
            }
        )
    df = df.rename(columns={"comments": "text", "teacher/course": "topic"})
    return df
"""),
    md("""
## 3. Heuristic baseline
"""),
    code("""
KEYWORDS = {
    "teacher": ["teacher", "sir", "madam", "prof"],
    "course": ["course", "syllabus", "lecture", "practical", "module"],
}


def keyword_rule(text: str):
    lower = text.lower()
    for label, words in KEYWORDS.items():
        if any(w in lower for w in words):
            return label
    return "course"
"""),
    md("""
## 4. TF–IDF baselines with explainability
"""),
    code("""
def run_topic_tfidf(train_df, val_df, analyzer="word"):
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )
    pipe.fit(train_df["text"], train_df["topic"])
    preds = pipe.predict(val_df["text"])
    print(classification_report(val_df["topic"], preds, zero_division=0))
    print(confusion_matrix(val_df["topic"], preds))

    explainer = LimeTextExplainer(class_names=pipe.classes_)
    explanation = explainer.explain_instance(val_df["text"].iloc[0], pipe.predict_proba, num_features=8)
    return pipe, explanation
"""),
    md("""
## 5. SBERT baseline
"""),
    code("""
def run_topic_sbert(train_df, val_df, model_name="all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not installed")

    model = SentenceTransformer(model_name)
    train_emb = model.encode(train_df["text"].tolist(), batch_size=16, show_progress_bar=True)
    val_emb = model.encode(val_df["text"].tolist(), batch_size=16, show_progress_bar=True)

    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(train_emb, train_df["topic"])
    preds = clf.predict(val_emb)
    print(classification_report(val_df["topic"], preds, zero_division=0))
    return clf, model
"""),
    md("""
## 6. Prompting baseline with rationale
"""),
    code("""
def prompt_topic(texts: List[str], model_name="gpt-4o-mini"):
    import openai

    client = openai.OpenAI()
    outputs = []
    for t in texts:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Decide if the comment is about a teacher or a course and respond with the label and a short justification."},
                {"role": "user", "content": t},
            ],
            temperature=0,
        )
        outputs.append(res.choices[0].message["content"])
    return outputs
"""),
    md("""
## 7. Robustness slice: sentiment-stratified
"""),
    code("""
def slice_by_sentiment(df: pd.DataFrame, sentiment_col="sentiment"):
    if sentiment_col not in df:
        return df
    return df.groupby(sentiment_col)["text"].apply(list)
"""),
    md("""
## 8. CLI entry point
"""),
    code("""
def main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Teacher vs course classification")
    parser.add_argument("--model", choices=["tfidf", "char", "sbert", "rule"], default="tfidf")
    args = parser.parse_args()

    df = load_topic_data()
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["topic"])

    if args.model == "rule":
        preds = [keyword_rule(t) for t in val_df["text"]]
        print(classification_report(val_df["topic"], preds, zero_division=0))
    elif args.model == "tfidf":
        run_topic_tfidf(train_df, val_df, analyzer="word")
    elif args.model == "char":
        run_topic_tfidf(train_df, val_df, analyzer="char")
    elif args.model == "sbert":
        run_topic_sbert(train_df, val_df)


if __name__ == "__main__":
    main_cli()
"""),
]

write_nb("teacher_course_classification.ipynb", teacher_course_cells)

print("Notebooks rebuilt.")
