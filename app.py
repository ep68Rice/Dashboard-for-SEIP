%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import simplejson

# Data Parsing Function (from your notebook)
def parse(filename):
  f = gzip.open(filename, 'rt')
  entry = {}
  for l in f:
    l = l.strip()
    colonPos = l.find(':')
    if colonPos == -1:
      yield entry
      entry = {}
      continue
    eName = l[:colonPos]
    rest = l[colonPos+2:]
    entry[eName] = rest
  yield entry

# --- Streamlit App --- #
st.title('Amazon Watches Review Analysis')

st.header('1. Data Loading and Preprocessing')

@st.cache_data # Cache data loading for performance
def load_data(filename):
    parsed_data = list(parse(filename))
    df = pd.DataFrame(parsed_data)

    # Cleaning (from your notebook)
    df = df.replace({'': np.nan, 'unknown': np.nan})
    df = df.dropna()

    # Ensure 'review/score' is numeric
    df['review/score'] = pd.to_numeric(df['review/score'], errors='coerce')

    # Ensure 'review/time' is datetime and extract year
    df['review/time'] = pd.to_datetime(df['review/time'], unit='s', errors='coerce')
    df['review_year'] = df['review/time'].dt.year

    # Calculate review lengths
    df['summary_length'] = df['review/summary'].apply(lambda x: len(str(x)))
    df['text_length'] = df['review/text'].apply(lambda x: len(str(x)))

    return df.dropna(subset=['review/score', 'review/time'])

with st.spinner('Loading and preprocessing data...'):
    df_app = load_data('Watches.txt.gz')
st.success('Data loaded and preprocessed!')

st.subheader('Raw Data Sample (First 5 Rows)')
st.dataframe(df_app.head())

st.subheader('DataFrame Information')
st.write(f"Shape of DataFrame: {df_app.shape}")
st.write(df_app.info(verbose=True, buf=io.StringIO())) # Capture info output


st.header('2. Review Score Analysis Over Time')
st.subheader('Average Review Score by Year')

average_scores_by_year = df_app.groupby('review_year')['review/score'].mean().reset_index()
fig_time, ax_time = plt.subplots(figsize=(10, 6))
ax_time.bar(average_scores_by_year['review_year'], average_scores_by_year['review/score'], color='skyblue')
ax_time.set_xlabel('Review Year')
ax_time.set_ylabel('Average Review Score')
ax_time.set_title('Average Review Score Over Time')
ax_time.tick_params(axis='x', rotation=45)
ax_time.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig_time)


st.header('3. Review Content Analysis')
st.subheader('Distribution of Review Summary and Text Lengths')

fig_lengths, axes_lengths = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(df_app['summary_length'], bins=30, kde=True, color='skyblue', ax=axes_lengths[0])
axes_lengths[0].set_title('Distribution of Review Summary Lengths')
axes_lengths[0].set_xlabel('Summary Length (Characters)')
axes_lengths[0].set_ylabel('Frequency')

sns.histplot(df_app['text_length'], bins=50, kde=True, color='lightcoral', ax=axes_lengths[1])
axes_lengths[1].set_title('Distribution of Review Text Lengths')
axes_lengths[1].set_xlabel('Text Length (Characters)')
axes_lengths[1].set_ylabel('Frequency')
plt.tight_layout()
st.pyplot(fig_lengths)


st.subheader('Distribution of Review Scores')
fig_score, ax_score = plt.subplots(figsize=(8, 5))
sns.histplot(df_app['review/score'], bins=5, kde=True, color='lightgreen', ax=ax_score)
ax_score.set_title('Distribution of Review Scores')
ax_score.set_xlabel('Review Score')
ax_score.set_ylabel('Number of Reviews')
ax_score.set_xticks([1, 2, 3, 4, 5])
st.pyplot(fig_score)


st.header('4. Product and User Participation')

st.subheader('Distribution of Product IDs')
product_counts = df_app['product/productId'].value_counts()
top_n = 20 # Can be adjusted

if len(product_counts) > top_n:
    top_products = product_counts.head(top_n)
    other_count = product_counts.iloc[top_n:].sum()
    plot_data_products = pd.concat([top_products, pd.Series({'Other': other_count})])
else:
    plot_data_products = product_counts

fig_products, ax_products = plt.subplots(figsize=(10, 10))
ax_products.pie(plot_data_products, labels=plot_data_products.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
ax_products.set_title(f'Distribution of Product IDs (Top {top_n} and Others)')
ax_products.axis('equal')
st.pyplot(fig_products)


st.subheader('Distribution of User Participation')
user_counts = df_app['review/userId'].value_counts()
top_n_users = 20 # Can be adjusted

if len(user_counts) > top_n_users:
    top_users = user_counts.head(top_n_users)
    other_user_count = user_counts.iloc[top_n_users:].sum()
    plot_data_users = pd.concat([top_users, pd.Series({'Other': other_user_count})])
else:
    plot_data_users = user_counts

fig_users, ax_users = plt.subplots(figsize=(10, 10))
ax_users.pie(plot_data_users, labels=plot_data_users.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
ax_users.set_title(f'Distribution of User Participation (Top {top_n_users} and Others)')
ax_users.axis('equal')
st.pyplot(fig_users)

