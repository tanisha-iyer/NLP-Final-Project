import pandas as pd
import streamlit as st
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from concurrent.futures import ThreadPoolExecutor
import os
from ragEngine import rag_answer, retrieve
from evaluation import GROUND_TRUTH, evaluate, rouge_l, bertscore, faithfulness_flag

# ── Configuration ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Reddit Climate Dashboard", layout="wide")

N_TOPICS = 10
N_KEYWORDS = 8
N_TOP_POSTS = 3
MAX_DOCS = 50_000
RECENT_WEEKS = 8       # weeks to consider "recent" for trending detection
TRENDING_THRESHOLD = 1.5   # recent_share must be X times overall_share to be trending

TOPIC_COLORS = [
    "#378ADD", "#1D9E75", "#E03206", "#7F77DD", "#ADAB0E",
    "#D4537E", "#639922", "#E24B4A", "#0F6E56", "#533AB7"
]

# Data Loading - loading simultanesouly to optimise 

@st.cache_data
def load_data():
    def load_posts():
        return pd.read_csv(
            r"C:\Users\Tanisha Iyer\Downloads\archive (4)\the-reddit-climate-change-dataset-posts.csv",
            usecols=["created_utc", "title", "score", "subreddit.name"]
        )
    def load_comments():
        return pd.read_csv(
            r"C:\Users\Tanisha Iyer\Downloads\archive (4)\the-reddit-climate-change-dataset-comments.csv",
            usecols=["created_utc", "body", "sentiment", "score", "subreddit.name"]
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_posts    = executor.submit(load_posts)
        f_comments = executor.submit(load_comments)
        posts    = f_posts.result()
        comments = f_comments.result()

    return posts, comments


@st.cache_data
def preprocess(posts, comments):
    posts["created_utc"] = pd.to_datetime(posts["created_utc"], unit="s")
    comments["created_utc"] = pd.to_datetime(comments["created_utc"], unit="s")

    posts.rename(columns={"subreddit.name": "subreddit"}, inplace=True)
    comments.rename(columns={"subreddit.name": "subreddit"}, inplace=True)

    comments["word_count"] = comments["body"].astype(str).str.split().str.len()

    posts.set_index("created_utc", inplace=True)
    comments.set_index("created_utc", inplace=True)

    return posts, comments


with st.spinner("Loading data..."):
    posts, comments = load_data()
    posts, comments = preprocess(posts, comments)

# Create the Sidebar 

st.sidebar.title("Filters")
time_range = st.sidebar.selectbox("Select Time Range", ["Daily", "Weekly", "Monthly"])

# ─KPI Metrics 

st.title("Reddit Climate Change Dashboard")

avg_sentiment = comments["sentiment"].mean() if "sentiment" in comments.columns else None

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Posts", f"{len(posts):,}")
col2.metric("Comments", f"{len(comments):,}")
col3.metric("Avg Post Score", f"{posts['score'].mean():.1f}")
col4.metric("Avg Comment Length", f"{comments['word_count'].mean():.1f} words")
if avg_sentiment is not None:
    col5.metric("Avg Sentiment", f"{avg_sentiment:.3f}", help="-1 = negative, +1 = positive")
else:
    col5.metric("Avg Sentiment", "Not Available")

# Time Stats 

@st.cache_data
def compute_time_series(posts, comments):
    return {
        "Daily":   (posts.resample("D").size(),  comments.resample("D").size()),
        "Weekly":  (posts.resample("W").size(),  comments.resample("W").size()),
        "Monthly": (posts.resample("ME").size(), comments.resample("ME").size()),
    }

time_data = compute_time_series(posts, comments)
posts_data, comments_data = time_data[time_range]

st.subheader(f"{time_range} Activity")
st.line_chart(pd.DataFrame({"Posts": posts_data, "Comments": comments_data}))

# Activity 

@st.cache_data
def compute_activity(comments):
    df = comments.copy()
    df["hour"] = df.index.hour
    df["day"]  = df.index.day_name()
    return df.groupby("hour").size(), df.groupby("day").size()

hourly_activity, day_activity = compute_activity(comments)

col_hour, col_day = st.columns(2)
with col_hour:
    st.subheader("Peak Activity by Hour")
    st.bar_chart(hourly_activity)
with col_day:
    st.subheader("Activity by Day of Week")
    st.bar_chart(day_activity)

# ── Subreddit Breakdown 

st.subheader("Top 15 Subreddits by Post Count")
st.bar_chart(posts["subreddit"].value_counts().head(15))

# ── Sentiment Distribution 

if "sentiment" in comments.columns:
    st.subheader("Sentiment Distribution")

    @st.cache_data
    def sentiment_dist(comments):
        bins   = [-1.01, -0.05, 0.05, 1.01]
        labels = ["Negative", "Neutral", "Positive"]
        return pd.cut(comments["sentiment"], bins=bins, labels=labels) \
                 .value_counts().reindex(labels)

    sent_counts = sentiment_dist(comments)
    fig_sent = px.bar(
        x=sent_counts.index, y=sent_counts.values,
        color=sent_counts.index,
        color_discrete_map={"Negative": "#E24B4A", "Neutral": "#888780", "Positive": "#1D9E75"},
        labels={"x": "Sentiment", "y": "Number of Comments"},
        title="Comments by Sentiment Category"
    )
    fig_sent.update_layout(showlegend=False)
    st.plotly_chart(fig_sent, use_container_width=True)

# ── Comment Length & Score 

with st.expander("Comment Length Distribution"):
    st.bar_chart(comments["word_count"].value_counts().head(50))

with st.expander("Post Score Distribution"):
    @st.cache_data
    def score_dist(posts):
        return posts[posts["score"] > 0]["score"].clip(upper=500).value_counts().sort_index()
    st.bar_chart(score_dist(posts))

# ── Text Cleaning 

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# LDA Topic 

@st.cache_resource
def fit_lda(posts_df):
    raw     = posts_df["title"].fillna("").reset_index(drop=True)
    sample  = raw.sample(min(MAX_DOCS, len(raw)), random_state=42)
    cleaned = sample.apply(clean_text)

    vectorizer = CountVectorizer(
        max_df=0.9, min_df=10, max_features=5000,
        stop_words="english", token_pattern=r"\b[a-z]{3,}\b"
    )
    dtm           = vectorizer.fit_transform(cleaned)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS, random_state=42,
        max_iter=15, learning_method="online"
    )
    doc_topic = lda.fit_transform(dtm)
    return lda, feature_names, doc_topic, sample.index


def build_topics(lda, feature_names, doc_topic):
    dominant = np.argmax(doc_topic, axis=1)
    topics   = []
    for i, comp in enumerate(lda.components_):
        top_idx  = comp.argsort()[-N_KEYWORDS:][::-1]
        keywords = [feature_names[j] for j in top_idx]
        share    = round((dominant == i).sum() / len(dominant) * 100, 1)
        topics.append({
            "id":       i,
            "label":    " / ".join(k.capitalize() for k in keywords[:3]),
            "keywords": keywords,
            "share":    share,
            "color":    TOPIC_COLORS[i % len(TOPIC_COLORS)]
        })
    return sorted(topics, key=lambda x: x["share"], reverse=True), dominant

#  Trending vs Persistent Topics  

@st.cache_data
def detect_trending(posts_df, _doc_topic, _sampled_idx, dominant, topics):
    """
    For each topic, compare its share in the most recent RECENT_WEEKS
    against its overall share. If recent_share >= TRENDING_THRESHOLD * overall_share
    it is tagged as Trending, otherwise Persistent.
    """
    posts_reset = posts_df.reset_index()   # brings created_utc back as a column
    cutoff      = posts_reset["created_utc"].max() - pd.Timedelta(weeks=RECENT_WEEKS)
    recent_mask = posts_reset["created_utc"] >= cutoff

    # Map sampled positions to the reset dataframe
    sampled_positions = np.arange(len(_sampled_idx))   # integer positions after reset

    recent_flags = recent_mask.iloc[_sampled_idx].values   # bool array, one per sample

    results = []
    for t in topics:
        topic_mask    = dominant == t["id"]
        overall_share = topic_mask.mean() * 100

        recent_in_topic  = (topic_mask & recent_flags).sum()
        recent_total     = recent_flags.sum()
        recent_share     = (recent_in_topic / recent_total * 100) if recent_total > 0 else 0

        ratio  = recent_share / overall_share if overall_share > 0 else 0
        status = "Trending" if ratio >= TRENDING_THRESHOLD else "Persistent"

        results.append({**t, "overall_share": overall_share,
                        "recent_share": round(recent_share, 1),
                        "trend_ratio":  round(ratio, 2),
                        "status":       status})
    return results

# ── Stance Detection ───────────────────────────────────────────────────────────

@st.cache_data
@st.cache_data
def assign_stances(comments_df, topics):
    # Sample for speed — 100k is plenty for representative stance analysis
    df = comments_df.sample(min(100_000, len(comments_df)), random_state=42)
    df = df.copy().reset_index()
    df["body_clean"] = df["body"].apply(clean_text)

    keyword_sets = {t["id"]: set(t["keywords"]) for t in topics}
    topic_labels = {t["id"]: t["label"] for t in topics}

    def find_topic(text):
        words  = set(text.split())
        scores = {tid: len(words & kws) for tid, kws in keyword_sets.items()}
        best   = max(scores, key=scores.get)
        return best if scores[best] > 0 else -1

    df["topic_id"]    = df["body_clean"].apply(find_topic)
    df["topic_label"] = df["topic_id"].map(topic_labels).fillna("Unassigned")
    df["stance"]      = df["sentiment"].apply(
        lambda s: "Support" if s >= 0.05 else ("Oppose" if s <= -0.05 else "Neutral")
    )
    return df


@st.cache_data
def extract_key_arguments(stance_df, topic_id, stance, n=3):
    """
    Extract the top N most representative comments for a given topic+stance
    using TF-IDF scoring — highest mean TF-IDF score = most "on-topic" comment.
    """
    subset = stance_df[(stance_df["topic_id"] == topic_id) &
                       (stance_df["stance"]   == stance)]["body_clean"]
    subset = subset[subset.str.len() > 20].dropna()

    if len(subset) < 2:
        return ["Not enough comments to summarise."]

    sample = subset.sample(min(500, len(subset)), random_state=42)
    try:
        tfidf  = TfidfVectorizer(stop_words="english", max_features=300)
        matrix = tfidf.fit_transform(sample)
        scores = np.asarray(matrix.mean(axis=1)).flatten()
        top_n  = scores.argsort()[-n:][::-1]
        return [subset.iloc[i][:250] for i in top_n]
    except Exception:
        return sample.head(n).tolist()


# ── Run Models ────────────────────────────────────────────────────────────────

st.header("Topic Analysis")
st.caption(f"LDA on up to {MAX_DOCS:,} post titles · {N_TOPICS} topics")

with st.spinner("Running topic model (cached after first run)..."):
    lda, feature_names, doc_topic, sampled_idx = fit_lda(posts)
    topics, dominant = build_topics(lda, feature_names, doc_topic)

with st.spinner("Detecting trending vs persistent topics..."):
    topics_tagged = detect_trending(posts, doc_topic, sampled_idx, dominant, topics)

with st.spinner("Assigning stances to comments..."):
    stance_df = assign_stances(comments, topics)

df_topics = pd.DataFrame([
    {"label": t["label"], "share": t["share"], "keywords": ", ".join(t["keywords"])}
    for t in topics
])

# ── Section 1 — Trending vs Persistent ────────────────────────────────────────

st.subheader("Trending vs Persistent Topics")
st.caption(
    f"Trending = topic's share in last {RECENT_WEEKS} weeks is "
    f"≥{TRENDING_THRESHOLD}× its overall share."
)

trending   = [t for t in topics_tagged if t["status"] == "Trending"]
persistent = [t for t in topics_tagged if t["status"] == "Persistent"]

def render_topic_badge(t):
    color  = "#E03206" if t["status"] == "Trending" else "#1D9E75"
    icon   = "↑ Trending" if t["status"] == "Trending" else "— Persistent"
    pills  = "".join(
        f'<span style="display:inline-block;background:#f0f2f6;color:#444;'
        f'font-size:11px;padding:1px 7px;border-radius:20px;margin:2px 2px 2px 0;">'
        f'{kw}</span>' for kw in t["keywords"]
    )
    return f"""
<div style="border:1px solid #ddd;border-radius:10px;padding:12px 14px;margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <span style="font-weight:600;font-size:14px;">{t['label']}</span>
    <span style="background:{color}22;color:{color};font-size:11px;
                 padding:2px 9px;border-radius:20px;font-weight:600;">{icon}</span>
  </div>
  <div style="font-size:12px;color:#666;margin-bottom:6px;">
    Overall: <b>{t['overall_share']:.1f}%</b> &nbsp;|&nbsp;
    Recent:  <b>{t['recent_share']}%</b> &nbsp;|&nbsp;
    Ratio:   <b>{t['trend_ratio']}×</b>
  </div>
  <div>{pills}</div>
</div>"""

col_tr, col_pe = st.columns(2)

with col_tr:
    st.markdown(f"#### Trending ({len(trending)})")
    if trending:
        for t in trending:
            st.markdown(render_topic_badge(t), unsafe_allow_html=True)
    else:
        st.info("No strongly trending topics detected in the recent window.")

with col_pe:
    st.markdown(f"#### Persistent ({len(persistent)})")
    for t in persistent:
        st.markdown(render_topic_badge(t), unsafe_allow_html=True)

# Trend ratio bar chart
df_tagged = pd.DataFrame(topics_tagged)
fig_trend = px.bar(
    df_tagged.sort_values("trend_ratio", ascending=True),
    x="trend_ratio", y="label", orientation="h",
    color="status",
    color_discrete_map={"Trending": "#E03206", "Persistent": "#1D9E75"},
    labels={"trend_ratio": "Recent/Overall Ratio", "label": "Topic", "status": "Status"},
    title=f"Topic Trend Ratio (threshold = {TRENDING_THRESHOLD}×)"
)
fig_trend.add_vline(x=TRENDING_THRESHOLD, line_dash="dash", line_color="gray",
                    annotation_text="Trending threshold")
fig_trend.update_layout(showlegend=True, yaxis_title="")
st.plotly_chart(fig_trend, use_container_width=True)

# ── Section 2 — Stance & Agreement Analysis ────────────────────────────────────

st.subheader("Agreement & Disagreement by Topic")
st.caption("Stance is derived from comment sentiment: Support ≥ 0.05 · Oppose ≤ −0.05 · Neutral in between.")

# Topic selector
topic_options = {t["label"]: t["id"] for t in topics}
selected_label = st.selectbox("Select a topic to analyse", list(topic_options.keys()))
selected_id    = topic_options[selected_label]

topic_comments = stance_df[stance_df["topic_id"] == selected_id]
stance_counts  = topic_comments["stance"].value_counts().reindex(
    ["Support", "Neutral", "Oppose"], fill_value=0
)
total = stance_counts.sum()

# Stance KPI row
s1, s2, s3, s4 = st.columns(4)
s1.metric("Comments on topic", f"{total:,}")
s2.metric("Support",  f"{stance_counts['Support']:,}",
          f"{stance_counts['Support']/total*100:.1f}%" if total else "0%")
s3.metric("Neutral",  f"{stance_counts['Neutral']:,}",
          f"{stance_counts['Neutral']/total*100:.1f}%" if total else "0%")
s4.metric("Oppose",   f"{stance_counts['Oppose']:,}",
          f"{stance_counts['Oppose']/total*100:.1f}%" if total else "0%")

# Stance donut chart
fig_donut = px.pie(
    names=stance_counts.index,
    values=stance_counts.values,
    hole=0.55,
    color=stance_counts.index,
    color_discrete_map={"Support": "#1D9E75", "Neutral": "#888780", "Oppose": "#E24B4A"},
    title=f"Stance distribution — {selected_label}"
)
fig_donut.update_traces(textinfo="percent+label")
fig_donut.update_layout(showlegend=False)
st.plotly_chart(fig_donut, use_container_width=True)

# Sentiment over time for this topic
st.markdown("##### Sentiment trend over time")
topic_time = (
    topic_comments.set_index("created_utc")["sentiment"]
    .resample("W").mean()
    .dropna()
    .reset_index()
)
if not topic_time.empty:
    fig_time = px.line(
        topic_time, x="created_utc", y="sentiment",
        labels={"created_utc": "Date", "sentiment": "Avg Sentiment"},
        title="Weekly average sentiment for selected topic"
    )
    fig_time.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_time, use_container_width=True)

# Key arguments per side
st.markdown("##### Key arguments by stance")
arg_col1, arg_col2 = st.columns(2)

with arg_col1:
    st.markdown("**Support**")
    args = extract_key_arguments(stance_df, selected_id, "Support")
    for i, a in enumerate(args, 1):
        st.markdown(
            f'<div style="background:#f0fdf4;border-left:3px solid #1D9E75;'
            f'padding:8px 12px;border-radius:4px;margin-bottom:8px;font-size:13px;">'
            f'<b>{i}.</b> {a}</div>',
            unsafe_allow_html=True
        )

with arg_col2:
    st.markdown("**Oppose**")
    args = extract_key_arguments(stance_df, selected_id, "Oppose")
    for i, a in enumerate(args, 1):
        st.markdown(
            f'<div style="background:#fff5f5;border-left:3px solid #E24B4A;'
            f'padding:8px 12px;border-radius:4px;margin-bottom:8px;font-size:13px;">'
            f'<b>{i}.</b> {a}</div>',
            unsafe_allow_html=True
        )

# Full stance breakdown table
with st.expander("Full stance breakdown across all topics"):
    breakdown = (
        stance_df[stance_df["topic_id"] >= 0]
        .groupby(["topic_label", "stance"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Support", "Neutral", "Oppose"], fill_value=0)
        .reset_index()
    )
    breakdown["Total"]       = breakdown[["Support", "Neutral", "Oppose"]].sum(axis=1)
    breakdown["Support %"]   = (breakdown["Support"] / breakdown["Total"] * 100).round(1)
    breakdown["Oppose %"]    = (breakdown["Oppose"]  / breakdown["Total"] * 100).round(1)
    breakdown["Controversy"] = (
        100 - abs(breakdown["Support %"] - breakdown["Oppose %"])
    ).round(1)
    st.dataframe(
        breakdown.sort_values("Controversy", ascending=False),
        use_container_width=True, hide_index=True
    )

# ── Original Topic Visualisation Tabs ─────────────────────────────────────────

st.header("Topic Deep Dive")

tab1, tab2, tab3, tab4 = st.tabs(["Bar Chart", "Topic Cards", "Explorer", "Treemap"])

with tab1:
    fig_bar = px.bar(
        df_topics, x="share", y="label", orientation="h",
        color="label", color_discrete_sequence=TOPIC_COLORS,
        labels={"share": "Share of posts (%)", "label": "Topic"},
        title="Topic Distribution"
    )
    fig_bar.update_layout(showlegend=False, xaxis_title="Share (%)", yaxis_title="")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("##### Keywords per topic")
    st.dataframe(df_topics[["label", "keywords", "share"]],
                 use_container_width=True, hide_index=True)

with tab2:
    cols = st.columns(2)
    for idx, t in enumerate(topics):
        with cols[idx % 2]:
            pills = "".join(
                f'<span style="display:inline-block;background:#f0f2f6;color:#333;'
                f'font-size:12px;padding:2px 8px;border-radius:20px;margin:2px 3px 2px 0;">'
                f'{kw}</span>' for kw in t["keywords"]
            )
            bar_w = min(int(t["share"] * 4), 100)
            st.markdown(f"""
<div style="border:1px solid #ddd;border-radius:10px;padding:14px 16px;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-weight:600;font-size:15px;">{t['label']}</span>
    <span style="background:{t['color']}22;color:{t['color']};font-size:12px;
                 padding:2px 10px;border-radius:20px;font-weight:600;">{t['share']}%</span>
  </div>
  <div style="margin:8px 0 4px;">{pills}</div>
  <div style="background:#eee;border-radius:4px;height:5px;margin-top:8px;">
    <div style="background:{t['color']};width:{bar_w}%;height:5px;border-radius:4px;"></div>
  </div>
</div>""", unsafe_allow_html=True)

with tab3:
    labels_list  = [f"{t['label']} ({t['share']}%)" for t in topics]
    chosen_label = st.selectbox("Select a topic", labels_list, key="explorer_select")
    chosen       = topics[labels_list.index(chosen_label)]

    st.markdown(f"**Share:** `{chosen['share']}%`")
    st.markdown("**Keywords:** " + " · ".join(f"`{k}`" for k in chosen["keywords"]))

    topic_mask   = dominant == chosen["id"]
    idx_in_sample = np.where(topic_mask)[0][:N_TOP_POSTS]
    post_indices  = sampled_idx[idx_in_sample]
    posts_reset   = posts.reset_index(drop=True)

    st.markdown("**Sample post titles:**")
    for i in post_indices:
        if i < len(posts_reset):
            st.markdown(f"> {str(posts_reset['title'].iloc[i])[:300]}")

with tab4:
    fig_tree = px.treemap(
        df_topics, path=["label"], values="share",
        color="share", color_continuous_scale="Blues",
        title="Topic Share Treemap"
    )
    fig_tree.update_traces(
        texttemplate="<b>%{label}</b><br>%{value}%",
        hovertemplate="<b>%{label}</b><br>Share: %{value}%<extra></extra>"
    )
    fig_tree.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_tree, use_container_width=True)

# RAG Section 

st.header("RAG Question Answering")
st.caption("Retrieval-Augmented Generation over Reddit climate change posts and comments.")

# API keys — store in st.secrets or sidebar inputs
st.sidebar.subheader("RAG API Keys")
groq_key = st.secrets.get("GROQ_API_KEY", "") or st.sidebar.text_input(
    "Groq API Key", type="password", key="groq_key"
)
api_keys = {"groq": groq_key}

# ── Query Interface ────────────────────────────────────────────────────────────

st.subheader("Ask a question")
query   = st.text_input("Enter your question about Reddit climate discussions")
llm     = st.selectbox("Choose LLM", ["Groq (LLaMA3)"])
top_k   = st.slider("Number of sources to retrieve", 3, 15, 8)

if st.button("Get Answer") and query:
    if not (groq_key or gemini_key):
        st.warning("Please enter an API key in the sidebar.")
    else:
        with st.spinner("Retrieving and generating..."):
            result = rag_answer(query, llm, api_keys, top_k=top_k)

        st.markdown("#### Answer")
        st.markdown(
            f'<div style="background:#f0f9ff;border-left:4px solid #378ADD;'
            f'padding:12px 16px;border-radius:6px;font-size:14px;">'
            f'{result["answer"]}</div>',
            unsafe_allow_html=True
        )

        with st.expander("Retrieved sources"):
            for i, c in enumerate(result["chunks"], 1):
                st.markdown(
                    f'<div style="background:#f8f9fa;border:1px solid #ddd;'
                    f'border-radius:6px;padding:8px 12px;margin-bottom:6px;font-size:12px;">'
                    f'<b>{i}. {c["type"].upper()}</b> · r/{c.get("subreddit","?")} · '
                    f'similarity: {c["score_sim"]}<br>{c["text"]}</div>',
                    unsafe_allow_html=True
                )

        with st.expander("Full prompt sent to LLM"):
            st.code(result["prompt"], language="text")


# ── Evaluation Section ─────────────────────────────────────────────────────────

st.subheader("Model Evaluation")
st.caption("Runs all 15 ground-truth questions through both LLMs and scores the answers.")

if st.button("Run Full Evaluation (takes ~2 min)"):
    if not (groq_key and gemini_key):
        st.warning("Both API keys required for comparative evaluation.")
    else:
        rows = []
        progress = st.progress(0)
        total    = len(GROUND_TRUTH) * 2

        for i, qa in enumerate(GROUND_TRUTH):
            for model_name in ["Groq (LLaMA3)", "Gemini"]:
                with st.spinner(f"Running {model_name} on Q{i+1}..."):
                    try:
                        result = rag_answer(qa["question"], model_name, api_keys)
                        rows.append({
                            "question":   qa["question"],
                            "reference":  qa["answer"],
                            "prediction": result["answer"],
                            "context":    result["context"],
                            "model":      model_name,
                            "type":       qa["type"],
                        })
                    except Exception as e:
                        rows.append({
                            "question":   qa["question"],
                            "reference":  qa["answer"],
                            "prediction": f"ERROR: {e}",
                            "context":    "",
                            "model":      model_name,
                            "type":       qa["type"],
                        })
                progress.progress((i * 2 + (0 if model_name == "Groq (LLaMA3)" else 1) + 1) / total)

        results_df = pd.DataFrame(rows)
        results_df = evaluate(results_df)

        # ── Summary table ──────────────────────────────────────────────────────
        st.markdown("#### Results by model")
        summary = (
            results_df[results_df["reference"] != "NOT_IN_CORPUS"]
            .groupby("model")
            .agg(
                ROUGE_L   =("rouge_l",      "mean"),
                BERTScore =("bertscore",    "mean"),
            )
            .round(4)
            .reset_index()
        )

        # Faithfulness: % of rows flagged as faithful (=1) — adversarial included
        for model_name in results_df["model"].unique():
            subset = results_df[results_df["model"] == model_name]
            flagged = (subset["faithfulness"] == 1).sum()
            auto_reviewable = (subset["faithfulness"] != -1).sum()
            pct = flagged / len(subset) * 100 if len(subset) else 0
            summary.loc[summary["model"] == model_name, "Faithfulness (auto %)"] = round(pct, 1)

        st.dataframe(summary, use_container_width=True, hide_index=True)

        # ── Per-question breakdown ─────────────────────────────────────────────
        st.markdown("#### Per-question breakdown")
        display_cols = ["question", "type", "model", "rouge_l", "bertscore",
                        "faithfulness", "prediction"]
        st.dataframe(
            results_df[display_cols].sort_values(["question", "model"]),
            use_container_width=True, hide_index=True
        )

        # ── Adversarial analysis ───────────────────────────────────────────────
        st.markdown("#### Adversarial question handling")
        adv = results_df[results_df["type"] == "adversarial"][
            ["question", "model", "prediction", "faithfulness"]
        ]
        st.dataframe(adv, use_container_width=True, hide_index=True)

        # ── Visual comparison ──────────────────────────────────────────────────
        fig_eval = px.bar(
            summary.melt(id_vars="model", value_vars=["ROUGE_L", "BERTScore"]),
            x="variable", y="value", color="model", barmode="group",
            color_discrete_map={"Groq (LLaMA3)": "#378ADD", "Gemini": "#1D9E75"},
            labels={"variable": "Metric", "value": "Score", "model": "LLM"},
            title="ROUGE-L and BERTScore by Model"
        )
        st.plotly_chart(fig_eval, use_container_width=True)

        # Save results for download
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download full results CSV", csv,
                           "rag_evaluation_results.csv", "text/csv")