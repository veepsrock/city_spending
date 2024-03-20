# Databricks notebook source
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic


# COMMAND ----------

# MAGIC %md
# MAGIC # Read in Data

# COMMAND ----------

df = pd.read_csv("Contracts_PrimeAwardSummaries_2024-02-20_H19M14S14_1.csv")
df = df[df["recipient_country_code"] == "USA"]

# COMMAND ----------

# MAGIC %md
# MAGIC # Build topic model

# COMMAND ----------

# Additional stop words
common_words = ["igf", "ot", "cl", "IGF::OT::IGF"]

# Create list of stop words
default_stop_words = set(TfidfVectorizer(stop_words="english").get_stop_words())
all_stop_words = list(default_stop_words.union(common_words))

# COMMAND ----------

vectorizer = TfidfVectorizer(stop_words=all_stop_words, ngram_range=(2,4), sublinear_tf=True)
sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# COMMAND ----------

topic_model = BERTopic(embedding_model=sentence_model, vectorizer_model=vectorizer, nr_topics=25)

# COMMAND ----------

topics, _ = topic_model.fit_transform(df['prime_award_base_transaction_description'].values)

# COMMAND ----------

topic_grams = []
for k in range(len(set(topics))):
    cur_top = topic_model.get_topic(k)
    if cur_top:
        cur_d = {'topic number': k}
        for j in range(5):
            cur_d[f'topic ngram {j+1}'] = cur_top[j][0]
        topic_grams.append(cur_d)
topics_df = pd.DataFrame(topic_grams)

# COMMAND ----------

topic_labels = topic_model.generate_topic_labels(nr_words=3,
                                                 topic_prefix=True,
                                                 separator=", ")

# COMMAND ----------

topic_names = pd.DataFrame([topic_model.topic_labels_])

# COMMAND ----------

topic_names.T.reset_index(names = ["topic", "topic_name" ])

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Add topic back to dataframe

# COMMAND ----------

df["topic"] = topics
df["topic"].value_counts()

# COMMAND ----------

topics_df

# COMMAND ----------

topic_model.visualize_barchart()


# COMMAND ----------

#df["product_or_service_code_description"].value_counts()

# COMMAND ----------

topic_model.visualize_topics()
