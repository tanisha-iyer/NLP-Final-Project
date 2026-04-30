"""
Evaluation: ROUGE-L, BERTScore, and faithfulness scoring.
"""

import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

# ── Ground truth Q&A set ───────────────────────────────────────────────────────

GROUND_TRUTH = [
    # Factual
    {
        "question": "Which subreddits are most active in climate change discussions?",
        "answer":   "The most active subreddits include r/climate, r/environment, r/climateskeptics, and r/worldnews.",
        "type":     "factual"
    },
    {
        "question": "What time of day do most climate-related Reddit comments get posted?",
        "answer":   "Most comments are posted during afternoon and evening hours in US time zones.",
        "type":     "factual"
    },
    {
        "question": "What is the general sentiment of Reddit comments on climate change?",
        "answer":   "The general sentiment leans slightly negative, reflecting concern and urgency about climate issues.",
        "type":     "factual"
    },
    {
        "question": "Are there posts about renewable energy on this subreddit?",
        "answer":   "Yes, renewable energy including solar and wind power is a frequently discussed topic.",
        "type":     "factual"
    },
    {
        "question": "What topics appear most frequently in climate change Reddit posts?",
        "answer":   "Common topics include global warming, policy, fossil fuels, renewable energy, and extreme weather.",
        "type":     "factual"
    },
    # Opinion
    {
        "question": "What do Reddit users think about government climate policy?",
        "answer":   "Users are generally critical of government inaction and call for stronger emissions regulations and international agreements.",
        "type":     "opinion"
    },
    {
        "question": "What do users think about electric vehicles as a climate solution?",
        "answer":   "Opinions are mixed. Many see EVs as a positive step but others argue they do not address systemic issues like energy production.",
        "type":     "opinion"
    },
    {
        "question": "How do Reddit users feel about climate scientists and their credibility?",
        "answer":   "Most users express trust in climate scientists and frustration with denial, though some subreddits contain skeptical viewpoints.",
        "type":     "opinion"
    },
    {
        "question": "What do users think about nuclear energy as a response to climate change?",
        "answer":   "Nuclear energy is a divisive topic. Supporters argue it is low-carbon and reliable while opponents cite safety and waste concerns.",
        "type":     "opinion"
    },
    {
        "question": "What are the most common arguments made by climate skeptics in the dataset?",
        "answer":   "Skeptics commonly argue that climate change is natural, that models are unreliable, or that economic costs of action are too high.",
        "type":     "opinion"
    },
    # Mixed
    {
        "question": "Have discussions about wildfires increased over time in the dataset?",
        "answer":   "Yes, wildfire-related posts increased notably around periods of major wildfire events.",
        "type":     "factual"
    },
    {
        "question": "What do users say about the link between climate change and extreme weather?",
        "answer":   "Most users accept the scientific link between climate change and more frequent extreme weather events.",
        "type":     "opinion"
    },
    {
        "question": "What is the Reddit community's view on carbon taxes?",
        "answer":   "Carbon taxes are generally supported as a market-based solution, though some users argue they disproportionately affect lower-income groups.",
        "type":     "opinion"
    },
    # adversial questions 
    {
        "question": "How does climate change impact marine biodiversity in the Pacific Ocean?",
        "answer":   "NOT_IN_CORPUS",
        "type":     "adversarial"
    },
    {
        "question": "How has sentiment on climate change changed since 2010?",
        "answer":   "NOT_IN_CORPUS",
        "type":     "adversarial"
    },
]

# scoring functions 

def rouge_l(prediction: str, reference: str) -> float:
    if reference == "NOT_IN_CORPUS":
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return round(scorer.score(reference, prediction)["rougeL"].fmeasure, 4)


def bertscore(predictions: list, references: list) -> list:
    valid = [(p, r) for p, r in zip(predictions, references) if r != "NOT_IN_CORPUS"]
    if not valid:
        return []
    preds, refs = zip(*valid)
    _, _, F1 = bert_score_fn(list(preds), list(refs), lang="en", verbose=False)
    return [round(f.item(), 4) for f in F1]


def faithfulness_flag(answer: str, context: str) -> int:
    refusal_phrases = [
        "i cannot find sufficient information in the dataset",  # exact prompt phrase
        "not present", "not in the", "cannot find", "no information",
        "not available", "i don't", "i do not", "not mentioned"
    ]
    if any(p in answer.lower() for p in refusal_phrases):
        return 1
    return -1   # -1 = needs manual review


def evaluate(results_df: pd.DataFrame) -> pd.DataFrame:
    results_df = results_df.copy()

    # ROUGE-L per row
    results_df["rouge_l"] = results_df.apply(
        lambda r: rouge_l(r["prediction"], r["reference"]), axis=1
    )

    # BERTScore per model
    bert_scores = {}
    for model_name in results_df["model"].unique():
        subset     = results_df[results_df["model"] == model_name]
        valid_mask = subset["reference"] != "NOT_IN_CORPUS"
        preds      = subset.loc[valid_mask, "prediction"].tolist()
        refs       = subset.loc[valid_mask, "reference"].tolist()
        if preds:
            scores = bertscore(preds, refs)
            for i, s in zip(subset.loc[valid_mask].index, scores):
                bert_scores[i] = s

    results_df["bertscore"] = results_df.index.map(lambda i: bert_scores.get(i, None))

    # Faithfulness
    results_df["faithfulness"] = results_df.apply(
        lambda r: faithfulness_flag(r["prediction"], r["context"]), axis=1
    )

    return results_df