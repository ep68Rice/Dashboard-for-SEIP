import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import simplejson


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

for e in parse("Watches.txt.gz"):
  print (simplejson.dumps(e))

import numpy as np
print(f"Original DataFrame shape: {df.shape}")

df = df.replace({'': np.nan, 'unknown': np.nan})

df= df.dropna()

# Ensure 'review/score' is numeric and 'review/time' is datetime
df['review/score'] = pd.to_numeric(df['review/score'], errors='coerce')
df['review/time'] = pd.to_datetime(df['review/time'], unit='s', errors='coerce')

# Extract the year from the review time
df['review_year'] = df['review/time'].dt.year

# Calculate the average review score for each year
average_scores_by_year = df.groupby('review_year')['review/score'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(average_scores_by_year['review_year'], average_scores_by_year['review/score'], color='skyblue')
plt.xlabel('Review Year')
plt.ylabel('Average Review Score')
plt.title('Average Review Score Over Time')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
