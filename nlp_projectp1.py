import pandas as pd 
import numpy as np 
import plotly.express as px
import streamlit as st 


""" Load Data """

@st.cache_data
def load_data():
    posts = pd.read_csv(r"C:\Users\Tanisha Iyer\Downloads\archive (4)\the-reddit-climate-change-dataset-posts.csv")
    comments = pd.read_csv(r"C:\Users\Tanisha Iyer\Downloads\archive (4)\the-reddit-climate-change-dataset-comments.csv")
    return posts, comments

posts, comments = load_data()

""" Preprocessing Data """

# Convert timestamps
posts['created_utc'] = pd.to_datetime(posts['created_utc'], unit='s')
comments['created_utc'] = pd.to_datetime(comments['created_utc'], unit='s')

# Comment length (word count)
comments['word_count'] = comments['body'].astype(str).apply(lambda x: len(x.split()))

st.title("Reddit Analytics Dashboard")

col1, col2, col3, col4 = st.columns(4)

num_posts = len(posts)
num_comments = len(comments)

# Only if author exists
num_users = comments['author'].nunique() if 'author' in comments.columns else "Not Available"

avg_comment_length = comments['word_count'].mean()

col1.metric("Posts", num_posts)
col2.metric("Comments", num_comments)
col3.metric("Users", num_users)
col4.metric("Avg Comment Length", f"{avg_comment_length:.2f} words")

"""Comments or Posts According to Time """

# Set index
posts.set_index('created_utc', inplace=True)
comments.set_index('created_utc', inplace=True)

# Resampling
posts_daily = posts.resample('D').size()
comments_daily = comments.resample('D').size()

posts_weekly = posts.resample('W').size()
comments_weekly = comments.resample('W').size()

posts_monthly = posts.resample('M').size()
comments_monthly = comments.resample('M').size()

#Dropdown 

time_range = st.selectbox(
    "Select According to Daily/Weekly/Monthly",
    ["Daily", "Weekly", "Monthly"]
)


if time_range == "Daily":
    posts_data = posts_daily
    comments_data = comments_daily

elif time_range == "Weekly":
    posts_data = posts_weekly
    comments_data = comments_weekly

else:  
    posts_data = posts_monthly
    comments_data = comments_monthly

# plot 

st.subheader(f"{time_range} Activity")

st.line_chart(pd.DataFrame({
    "Posts": posts_data,
    "Comments": comments_data
}))

""" Streamlit Plotting """

st.subheader("Daily Activity")
st.line_chart(pd.DataFrame({
    "Posts": posts_daily,
    "Comments": comments_daily
}))

""" Peak Activity Time """

comments['hour'] = comments.index.hour

hourly_activity = comments.groupby('hour').size()

st.subheader("Peak Activity by Hour")
st.bar_chart(hourly_activity)

"""Peak Activity Day of the Week """

comments['day'] = comments.index.day_name()

day_activity = comments.groupby('day').size()

st.subheader("Activity by Day of Week")
st.bar_chart(day_activity)

""" Comment Length Distribution """

st.subheader("Distribution of Comment Length")
st.histogram = st.bar_chart(comments['word_count'].value_counts().head(50))

""" Comments per Post """

comments_per_post = comments.groupby('link_id').size()

st.subheader("Comments per Post (Avg)")
st.write(comments_per_post.mean())

st.sidebar.title("Filters")
time_range = st.sidebar.selectbox("Select Range", ["Daily", "Weekly", "Monthly"])

""" Statistics """ 
unique_users = df1["id"].nunique()
posts = df1["type"].count()
comments = df2["type"].count()



