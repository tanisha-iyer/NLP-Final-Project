import pandas as pd
import streamlit as st

""" Load Data """

@st.cache_data
def load_data():
    posts = pd.read_csv(r"C:\Users\Tanisha Iyer\Downloads\archive (4)\the-reddit-climate-change-dataset-posts.csv")
    comments = pd.read_csv(r"C:\Users\Tanisha Iyer\Downloads\archive (4)\the-reddit-climate-change-dataset-comments.csv")
    return posts, comments

posts, comments = load_data()

""" Preprocessing """

# Convert timestamps
posts['created_utc'] = pd.to_datetime(posts['created_utc'], unit='s')
comments['created_utc'] = pd.to_datetime(comments['created_utc'], unit='s')

# Comment length
comments['word_count'] = comments['body'].astype(str).str.split().str.len()

# Set index
posts.set_index('created_utc', inplace=True)
comments.set_index('created_utc', inplace=True)

""" Sidebar """

st.sidebar.title("Filters")
time_range = st.sidebar.selectbox(
    "Select Time Range",
    ["Daily", "Weekly", "Monthly"]
)

""" KPI Metrics """

st.title("Reddit Analytics Dashboard")

num_posts = len(posts)
num_comments = len(comments)
avg_comment_length = comments['word_count'].mean()

# ✅ Sentiment Metric
if 'sentiment' in comments.columns:
    avg_sentiment = comments['sentiment'].mean()
else:
    avg_sentiment = None

col1, col2, col3, col4 = st.columns(4)

col1.metric("Posts", f"{num_posts:,}")
col2.metric("Comments", f"{num_comments:,}")
col3.metric("Avg Comment Length", f"{avg_comment_length:.2f} words")

if avg_sentiment is not None:
    col4.metric(
        "Avg Sentiment",
        f"{avg_sentiment:.3f}",
        help="-1 = negative, +1 = positive"
    )
else:
    col4.metric("Avg Sentiment", "Not Available")

""" Time-Based Analysis """

# Resampling
posts_daily = posts.resample('D').size()
comments_daily = comments.resample('D').size()

posts_weekly = posts.resample('W').size()
comments_weekly = comments.resample('W').size()

posts_monthly = posts.resample('ME').size()
comments_monthly = comments.resample('ME').size()

# Dynamic selection
if time_range == "Daily":
    posts_data = posts_daily
    comments_data = comments_daily
elif time_range == "Weekly":
    posts_data = posts_weekly
    comments_data = comments_weekly
else:
    posts_data = posts_monthly
    comments_data = comments_monthly

st.subheader(f"{time_range} Activity")

st.line_chart(pd.DataFrame({
    "Posts": posts_data,
    "Comments": comments_data
}))

""" Peak Activity """

comments['hour'] = comments.index.hour
hourly_activity = comments.groupby('hour').size()

st.subheader("Peak Activity by Hour")
st.bar_chart(hourly_activity)

""" Day of Week """

comments['day'] = comments.index.day_name()
day_activity = comments.groupby('day').size()

st.subheader("Activity by Day of Week")
st.bar_chart(day_activity)

""" Comment Length Distribution """

st.subheader("Comment Length Distribution")
st.bar_chart(comments['word_count'].value_counts().head(50))

""" Engagement: Comments per Post """

if 'link_id' in comments.columns:
    comments_per_thread = comments.groupby('link_id').size()

    st.subheader("Comments per Post (Average)")
    st.write(f"{comments_per_thread.mean():.2f}")

    st.subheader("Comments per Post Distribution")
    st.bar_chart(comments_per_thread.value_counts().head(50))

    st.subheader("Top 10 Most Active Threads")
    top_threads = comments_per_thread.sort_values(ascending=False).head(10)
    st.bar_chart(top_threads)
else:
    st.warning("link_id not found: cannot compute comments per post")