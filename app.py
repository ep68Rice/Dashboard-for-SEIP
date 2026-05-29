import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import simplejson

# ── Data Parsing ────────────────────────────────────────────────────────────────

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

# ── Data Loading ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(filename):
    parsed_data = list(parse(filename))
    df = pd.DataFrame(parsed_data)

    # Clean
    df = df.replace({'': np.nan, 'unknown': np.nan})
    df = df.dropna()

    # Types
    df['review/score'] = pd.to_numeric(df['review/score'], errors='coerce')

    def parse_time(val):
        try:
            return pd.to_datetime(int(val), unit='s')
        except:
            return pd.to_datetime(val, errors='coerce')

    df['review/time'] = df['review/time'].apply(parse_time)
    df['review_year'] = df['review/time'].dt.year

    # Feature engineering
    df['summary_length'] = df['review/summary'].apply(lambda x: len(str(x)))
    df['text_length'] = df['review/text'].apply(lambda x: len(str(x)))

    return df.dropna(subset=['review/score', 'review/time'])

# ── App ──────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='Amazon Watches Reviews', layout='wide')
st.title('🕐 Amazon Watches Review Analysis')

with st.spinner('Loading data...'):
    df = load_data('Watches.txt.gz')

st.success(f"Loaded {len(df):,} reviews")

# ── 1. Average Review Score Over Time ───────────────────────────────────────────

st.header('📅 Review Score Over Time')

avg_by_year = df.groupby('review_year')['review/score'].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(avg_by_year['review_year'], avg_by_year['review/score'], color='skyblue')
ax.set_xlabel('Year')
ax.set_ylabel('Average Review Score')
ax.set_title('Average Review Score Over Time')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(fig)

# ── 2. Review Length Distributions ──────────────────────────────────────────────

st.header('📝 Review Content Analysis')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df['summary_length'], bins=30, kde=True, color='skyblue', ax=axes[0])
axes[0].set_title('Distribution of Summary Lengths')
axes[0].set_xlabel('Characters')
axes[0].set_ylabel('Frequency')

sns.histplot(df['text_length'], bins=50, kde=True, color='lightcoral', ax=axes[1])
axes[1].set_title('Distribution of Review Text Lengths')
axes[1].set_xlabel('Characters')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
st.pyplot(fig)

# ── 3. Score Distribution ────────────────────────────────────────────────────────

st.header('⭐ Score Distribution')

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['review/score'], bins=5, kde=True, color='lightgreen', ax=ax)
ax.set_title('Distribution of Review Scores')
ax.set_xlabel('Review Score')
ax.set_ylabel('Number of Reviews')
ax.set_xticks([1, 2, 3, 4, 5])
plt.tight_layout()
st.pyplot(fig)

# ── 4. Top Products ──────────────────────────────────────────────────────────────

st.header('📦 Product & User Participation')

TOP_N = 20

product_counts = df['product/productId'].value_counts()
top_products = product_counts.head(TOP_N)
if len(product_counts) > TOP_N:
    top_products['Other'] = product_counts.iloc[TOP_N:].sum()

fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(top_products, labels=top_products.index, autopct='%1.1f%%',
       startangle=140, pctdistance=0.85)
ax.set_title(f'Top {TOP_N} Products by Review Count')
ax.axis('equal')
st.pyplot(fig)

# ── 5. Top Users ─────────────────────────────────────────────────────────────────

user_counts = df['review/userId'].value_counts()
top_users = user_counts.head(TOP_N)
if len(user_counts) > TOP_N:
    top_users['Other'] = user_counts.iloc[TOP_N:].sum()

fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(top_users, labels=top_users.index, autopct='%1.1f%%',
       startangle=140, pctdistance=0.85)
ax.set_title(f'Top {TOP_N} Users by Review Count')
ax.axis('equal')
st.pyplot(fig)
