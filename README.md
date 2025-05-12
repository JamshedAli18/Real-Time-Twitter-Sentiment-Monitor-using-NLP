# Real-Time Twitter Sentiment Monitor using NLP

## Overview
This project implements a sentiment analysis system to classify X posts (tweets) as Positive, Negative, or Neutral using Natural Language Processing (NLP). Built with a Kaggle dataset, it showcases text preprocessing, exploratory data analysis (EDA), and machine learning with a Logistic Regression model. The project is designed to monitor social media sentiment and is ideal for demonstrating NLP and data science skills.
 
## Dataset
- **Name**: [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- **Source**: Kaggle
- **Description**: Contains 74,681 tweets with labeled sentiments:
  - Columns: `Tweet_ID`, `entity`, `sentiment` (Positive, Negative, Neutral), `Tweet_content`
- **Use Case**: Multi-class classification to predict sentiment from tweet text.

## Project Steps
1. **Exploratory Data Analysis (EDA)**:
   - Visualized sentiment distribution using a count plot.
   - Generated word clouds for Positive and Negative tweets to highlight frequent words.
2. **Data Preprocessing**:
   - Cleaned text by removing URLs, special characters, and stopwords.
   - Tokenized and vectorized text using CountVectorizer.
   - Split data into 80% training and 20% testing sets.
3. **Model Training**:
   - Trained a Logistic Regression model for sentiment classification.
4. **Model Evaluation**:
   - Evaluated using accuracy, precision, recall, F1-score, and confusion matrix.
5. **Results**:
   - **Accuracy**: 0.70
   - **Precision**: 0.71
   - **Recall**: 0.70
   - **F1-Score**: 0.70


## Key Findings
- Positive tweets often contain words like "great," "love," and "awesome."
- Negative tweets frequently include "bad," "issue," and "problem."
- The Logistic Regression model achieves 70% accuracy, with balanced precision and recall, indicating decent performance for a baseline NLP model.

## Tools Used
- **Python Libraries**: Pandas, NLTK, Scikit-learn, Seaborn, Matplotlib, WordCloud
- **Environment**: Kaggle Notebook

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jamshedali18/twitter-sentiment-monitor.git
   ```
2. **Run on Kaggle**:
   - Open the [Kaggle Notebook](https://www.kaggle.com/your-username/twitter-sentiment-analysis-notebook).
   - Add the [Twitter Entity Sentiment Analysis Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) to your notebook.
   - Copy the code from `Twitter_Sentiment_Analysis.ipynb` and run all cells.
3. **Local Setup** (Optional):
   - Install dependencies:
     ```bash
     pip install pandas nltk scikit-learn seaborn matplotlib wordcloud
     ```
   - Download the dataset CSV and update the file path in the notebook.
   - Run the Jupyter Notebook locally:
     ```bash
     jupyter notebook Twitter_Sentiment_Analysis.ipynb
     ```


## License
This project is licensed under the MIT License.

*This project demonstrates skills in NLP, text preprocessing, and machine learning for social media analysis.*
