# British Airways Review Sentiment Analysis with BERT and SEMMA Research Framework

## 1. Project Overview

This project implements a complete Natural Language Processing (NLP) pipeline to perform sentiment analysis on customer reviews for British Airways. It follows the **SEMMA (Sample, Explore, Modify, Model, Assess)** data mining framework to systematically gather, clean, model, and evaluate data.

The primary goal is to collect real-world reviews, process the unstructured text, and train a Transformer-based model (DistilBERT) to classify each review as **positive**, **negative**, or **neutral**.

---

## 2. Key Features

* **Web Scraping**: Dynamically scrapes 1,000 reviews from `airlinequality.com`.
* **Data Cleaning**: Advanced data cleaning and preprocessing, including regex for text sanitization, handling of complex date formats, and imputation of missing values.
* **Hybrid Sentiment Labeling**: Uses a powerful `distilbert-base-uncased-finetuned-sst-2-english` model for initial labeling, with `VADER` as a fallback for neutral or low-confidence predictions.
* **Model Training**: Fine-tunes a `distilbert-base-uncased` model for 3-class sentiment classification using the Hugging Face `Trainer` API.
* **Experiment Tracking**: Integrates with `wandb` (Weights & Biases) to log and monitor training metrics.
* **Evaluation**: Provides a detailed classification report and visualizations, including word clouds for positive vs. negative sentiment.

---

## 3. Methodology (SEMMA Framework)

This project is structured around the 5 stages of the SEMMA framework.

### 1. Sample
* **Source**: `https://www.airlinequality.com/airline-reviews/british-airways/`
* **Method**: Used `requests` to fetch the HTML and `BeautifulSoup` to parse and extract data.
* **Data Collected**: 1,000 reviews, including user name, date, review text, and various star ratings (e.g., `seat_comfort_star`, `cabin_staff_service_star`, etc.).

### 2. Explore
* Initial analysis with `pandas` revealed several data quality issues:
    * Significant number of `NaN` (missing) values across all rating columns.
    * The `seat_comfort_value` column was 100% empty and was dropped.
    * The `wifi_and_connectivity_star` column had over 67% missing values and was dropped.
    * The review date was improperly formatted and merged with the `user_name` string.

### 3. Modify
* **Date Cleaning**: The review date was extracted from the `user_name` column using regular expressions (`re`).
* **Handling Missing Values (Imputation)**:
    * **Categorical**: `aircraft_value` `NaNs` were filled using the **mode** (the most frequent aircraft, `A320`).
    * **Numerical**: All star-rating `NaNs` (e.g., `seat_comfort_star`, `ground_service_star`) were filled using the **median** value of their respective columns.
    * Rows with `NaN` in `type_of_traveller_value` (2 rows) were dropped.
* **Text Cleaning**: Prefixes like "✅Trip Verified|" and "Not Verified|" were removed from the `text_content` to isolate the review text.
* **Sentiment Labeling**: A new target column, `sentiment`, was created. The `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face was used to predict 'positive' or 'negative'. A confidence threshold of 0.85 was set; any prediction below this was classified as 'neutral'. This approach was supplemented by `VADER` as a fallback.
* **Label Encoding**: The string labels ('negative', 'neutral', 'positive') were mapped to integers (0, 1, 2) for modeling.

### 4. Model
* **Data Split**: The labeled dataset (998 rows) was split into training (70%), validation (15%), and test (15%) sets, stratified by the `label` column.
* **Tokenization**: Text was tokenized using `AutoTokenizer` from `distilbert-base-uncased`, with padding and truncation set to `max_length=128`.
* **Model**: A `AutoModelForSequenceClassification` with a `distilbert-base-uncased` base was initialized for 3 labels.
* **Training**: The model was fine-tuned for 3 epochs using the `Trainer` API. Training arguments included:
    * `learning_rate`: 2e-5
    * `per_device_train_batch_size`: 16
    * `per_device_eval_batch_size`: 32
    * `weight_decay`: 0.01
    * `evaluation_strategy`: "epoch"
    * `load_best_model_at_end`: True (based on `f1` score)

### 5. Assess
* The model's performance was evaluated on the unseen test set.
* Visualizations (Pie Chart, Word Clouds) were generated to understand the final sentiment distribution and key terms.

---

## 4. Results & Evaluation

### Model Performance (Test Set)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 90.67% |
| **F1-Score (Macro)** | 0.60 |

The classification report shows high performance for the **negative** (F1: 0.95) and **positive** (F1: 0.84) classes. The macro F1-score is lower due to the highly imbalanced and small sample size of the **neutral** class (only 6 samples in the test set).

### Sentiment Distribution
* **Negative**: 74.05%
* **Positive**: 22.44%
* **Neutral**: 3.51%

### Dominant Words
* **Positive Reviews**: Key terms included "good", "flight", "seat", "service", "time", and "food".
* **Negative Reviews**: Key terms included "flight", "ba", "service", "london", "time", "staff", "hour", and "check".

---

## 5. Technologies Used

* **Data Collection**: `requests`, `beautifulsoup4`
* **Data Manipulation**: `pandas`, `numpy`, `re` (Regex)
* **NLP & Modeling**: `transformers`, `torch`, `datasets`, `scikit-learn`
* **Sentiment Labeling**: `vaderSentiment`
* **Visualization**: `matplotlib`, `wordcloud`
* **Experiment Tracking**: `wandb`
* **Utilities**: `nltk` (for stopwords in word cloud)
