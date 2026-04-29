import pandas as pd 
import numpy as np 
import plotly.express as px

""" Import data """
df1 = pd.read_csv(r"C:\Users\Tanisha Iyer\Downloads\archive (3)\the-antiwork-subreddit-dataset-posts.csv")
df2 = pd.read_csv(r"C:\Users\Tanisha Iyer\Downloads\archive (3)\the-antiwork-subreddit-dataset-comments.csv")

""" Statistics """ 
unique_users = df1["id"].nunique()
posts = df1["type"].count()
comments = df2["type"].count()

