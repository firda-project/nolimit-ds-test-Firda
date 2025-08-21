# Mental-Health-Classification

## Overview
This project tackles **multi-class text classification** in the mental health domain.  
The dataset contains short user posts labeled into 7 categories:
- **normal**
- **depression**
- **suicidal**
- **bipolar**
- **anxiety**
- **stress**
- **personality disorder**

Dataset: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data

### Test Requirements Covered
- ✅ **Use Hugging Face models** → SBERT (`all-MiniLM-L6-v2`) from `sentence-transformers`.
- ✅ **Use embeddings for representation/search** → SBERT embeddings used as classifier features and for semantic search with **FAISS**.
- ✅ **Compare 3 methods** → Logistic Regression, Linear SVM, XGBoost (all trained on SBERT embeddings).
- ✅ **EDA included** → class distribution, token length, sample posts.
- ✅ **Submission** → predictions generated for `sample_input.csv` (id,text) into `submission.csv`.
- ✅ **Small sample dataset included** → `sample_input.csv` with 10 demo rows.

---

## Pipeline
1. **EDA**
   - Class distribution
   - Token length histogram
   - Example posts per label
2. **Preprocessing**
   - Lowercasing, remove URL, mentions, emojis
   - No stemming/lemmatization (SBERT handles tokenization)
3. **Train/Validation split (stratified)**
4. **Embedding**  
   - Encode posts with SBERT → dense vectors (384 dim)
5. **Classification models**
   - Logistic Regression
   - Linear SVM
   - XGBoost
6. **Evaluation**
   - Metrics: Accuracy, Macro-F1
   - Confusion Matrix
   - Select best model
7. **Semantic Search**
   - Build FAISS index over embeddings
   - Query similar posts
8. **Submission**
   - Input: `sample_input.csv` (id,text; no label)
   - Output: `submission.csv` (id,label)

---

## Flowchart
<img width="850" height="385" alt="image" src="https://github.com/user-attachments/assets/e681ae5c-cacc-4169-a3ef-9231fa406bb0" />

Source:https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FFlow-chart-for-machine-learning-workflow_fig1_342778782&psig=AOvVaw024vSCZp_T3v8xL9owpNTT&ust=1755831766336000&source=images&cd=vfe&opi=89978449&ved=0CBgQjhxqFwoTCMjvvpf1mo8DFQAAAAAdAAAAABAK

---

## How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
