# BERT-for-paraphrase-detection
BERT for paraphrase detection

Group JuHa:
Hana Dubovská - hadu@itu.dk
Julia Trznadel - jtrz@itu.dk

# Problem
Given two questions from Quora, we obtain two questions as a pair, which includes a binary labels (0-not a paraphrase, 1-paraphrase). Duplicate question detection saves time of the users and keeps Q&A platforms clean.

# Data

We are using glue/qqp via HuggingFaceDatasets. Through data-exploration we found out that around 63% is labeled as not a paraphrase and 37% as a paraphrase. Quite imbalanced, which matters for threshold tuning. Lenght of the questions is mostly short (around 10-15 words), but occasionally it exeeeds 50+ tokens. Maximum lenght of 128 tokens covers the vast majority of pairs.
