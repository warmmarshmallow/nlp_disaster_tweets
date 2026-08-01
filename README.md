# Kaggle NLP with Disaster Tweets

This repository contains my solution for the Kaggle Natural Language Processing with Disaster Tweets competition, where the goal is to classify whether a tweet describes a real disaster or not.

Competition Link: https://www.kaggle.com/competitions/nlp-getting-started

## Results

| Model | Public Leaderboard |
|-------|-------------------:|
| TF-IDF + Logistic Regression | **0.78853** |

## Project Structure

```
.
├── project.ipynb
├── train.csv
├── test.csv
├── submission.csv
└── README.md
```

## Exploratory Data Analysis

Before training the model, I explored the dataset to better understand the characteristics of the tweets and the target labels.

The analysis includes:

- Distribution of disaster vs. non-disaster tweets
- Inspection of tweet length
- Common words and phrases
- Class balance analysis

## Data Preprocessing

The preprocessing pipeline consists of:

- Splitting the training data into training and validation sets
- Converting tweet text into TF-IDF feature vectors
- Using unigram and bigram features
- Learning the TF-IDF vocabulary only from the training split to prevent data leakage

## Model

The model is implemented using Scikit-Learn's **Logistic Regression**.

Training pipeline:

1. Split the labeled training data into training and validation sets
2. Vectorize tweet text using TF-IDF
3. Train a Logistic Regression classifier
4. Evaluate the model using the validation set
5. Generate predictions for the Kaggle test dataset
6. Export predictions as `submission.csv`

### Model Parameters

**TF-IDF Vectorizer**

- Unigrams and bigrams (`ngram_range=(1,2)`)
- Minimum document frequency of 2 (`min_df=2`)
- Sublinear term frequency scaling (`sublinear_tf=True`)

**Logistic Regression**

- `C = 2.0`
- `max_iter = 1000`
- `class_weight = "balanced"`

## Technologies Used

- Python
- Pandas
- Scikit-Learn

## Future Improvements

Possible improvements include:

- Additional text preprocessing
- Stop-word removal
- Lemmatization and stemming
- Hyperparameter tuning
- Linear Support Vector Machines (SVM)
- Naive Bayes
- Transformer models (BERT, RoBERTa, DistilBERT)
- Cross-validation
- Ensemble methods

## Lessons Learned

Through this project I gained practical experience with:

- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Text Feature Engineering
- Logistic Regression
- Binary Text Classification
- Model Evaluation using F1 Score
- Kaggle competition workflow
- Building end-to-end machine learning pipelines
