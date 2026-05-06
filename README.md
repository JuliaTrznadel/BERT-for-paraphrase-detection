# BERT-for-paraphrase-detection
BERT for paraphrase detection

Group JuHa:
Hana Dubovská - hadu@itu.dk
Julia Trznadel - jtrz@itu.dk

# Problem
Given two questions from Quora, we obtain two questions as a pair, which includes a binary labels (0-not a paraphrase, 1-paraphrase). Duplicate question detection saves time of the users and keeps Q&A platforms clean.

# Data

We are using glue/qqp via HuggingFaceDatasets. Through data exploration we found out that around 63% is labeled as not a paraphrase and 37% as a paraphrase. Quite imbalanced, which matters for threshold tuning. Lenght of the questions is mostly short (around 10 words).

Train (full set) - around 364k pairs
Train (used) - 10 000 pairs
Validation (used) - 3 000 pairs

# Approach 

## TF-IDF and Logistic Regression
Before BERT we built a simple baseline. Starting with concatinating question1 [SEP] question2 into one string. Then we vectorize with TF-IDF (unigrams and biagrams, top 100k features) and classification with logistic regression (class_weight="balanced")to handle imbalance. We trained it on 50k pairs. 

## Main model: Fine-tuned BERT

Training setup:
Base mode: bert-base-uncased
Max sequence length: 96 tokens
Batch size: 32
Learning rate: 2e-5
Epochs: 2
Weight decay: 0.01
Warmup ratio: 10%
Best model selection: max F1 on validation

Threshold Tuning:
After training, instead of hard-coding threshold=0.5, we changed thresholds from 0.1 to 0.9 and pick the one that maximises F1 on the validation set. This matters because the classes are imbalanced.

# Results
All BERT results are on the 3k validation subset. Training was limited to 10k examples due to compute constraints, if we would use full dataset, we suspect the results would be higher. TF-IDF uses the full validation set (40,430 examples) after training on 50k pairs.


TF-IDF + Logistic Regression (50k train, full val set 40,430 examples)
 
| Class | Precision | Recall | F1 |
|---|---|---|---|
| not_dup | 0.8061 | 0.7938 | 0.7999 | 
| duplicate | 0.6551 | 0.6723 | 0.6636 | 
| macro avg | 0.7306 | 0.7330 | 0.7317 | 
| weighted avg | 0.7505 | 0.7490 | 0.7497 | 

BERT (10k train, 3k val subset, threshold=0.5)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| not_dup | 0.9195 | 0.7992 | 0.8551 |
| duplicate | 0.7213 | 0.8814 | 0.7934 |
| macro avg | 0.8204 | 0.8403 | 0.8242 |
| weighted avg | 0.8460 | 0.8297 | 0.8322 |

### Summary comparison

| Model | Train size | Accuracy | Macro F1 |
|---|---|---|---|
| TF-IDF + LogReg | 50k | 0.749 | 0.732 |
| BERT (thr=0.50) | 10k | 0.830 | 0.824 |
| BERT (thr=0.45) | 10k | — | **0.7943** (dup only) |

BERT trained on 5× fewer examples still beats TF-IDF by +8.3 pp accuracy and +9.2 pp macro F1.

### Threshold tuning

Best threshold is 0.45 (F1 on duplicate class=0.7943, marginally better than 0.50).
 
This reveals an interesting asymmetry. F1 stays flat between 0.35–0.65 (within ~0.007 of each other), then drops sharply above 0.75 as recall collapses. This means the model is quite performing. We didn't gain much from aggressive threshold tuning, but lowering slightly to 0.45 helps because the duplicate class is the minority class and the model is slightly overconfident.


# Error analysis

We also checked false positives and false negatives. Where:

- False positive: model cried "duplicate" but wasn't
- false negatives: missed actual duplicates

Questions like "anxiety and depression" vs "anxiety, loneliness and depression" getting 0.877 confidence. Furthermore, "stock market" vs "sensex" (0.008 confidence!), or a typo "auora" is not recognised by the model. These are all limitations. BERT operates on subword tokens and doesn't have lookup-style world knowledge, it doesn't know that sensex is a stock market index. 

# Discussion

Despite training on only 10k examples (2.7% of the full training set), BERT outperforms the TF-IDF baseline trained on 50k examples. It leads to a conclusion that pre-trained representations transfer extremely well to tasks involving paraphrase detection. Although, 10k training examples is a real limitation, keeping in mind that the full QQP training set has 364k pairs.

The classes are somewhat imbalanced (63%/37%) and we didn't address it for BERT. The TF-IDF baseline used class_weight="balanced", but the BERT didn't. Adding class weights or sampling duplicates during fine-tuning would likely boost the performance.

Threshold tuning: Threshold tuning is worth doing, but the F1 curve is quite flat in the 0.35–0.65 range, suggesting the model is well calibrated. At 0.45 only improves F1 by ~0.001 over 0.50.

# Note

AI tools were used as a coding support for this project
