# Financial Market Analysis

## Overview
This project focuses on sentiment analysis of financial market news using machine learning techniques. The primary objective is to classify news headlines as positive or negative based on sentiment. By analyzing financial news, investors and analysts can make more informed decisions regarding market trends and potential risks.

## Dataset
The dataset is sourced from the [YBI-Foundation Dataset](https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Financial%20Market%20News.csv). It contains financial news headlines along with sentiment labels. The dataset includes multiple columns representing different aspects of financial news, which are preprocessed and merged into a single text feature for analysis.

## Project Workflow
1. **Import Libraries**: Load required Python libraries like Pandas, NumPy, and Scikit-learn.
2. **Data Loading**: Read the dataset from a CSV file and inspect its structure.
3. **Exploratory Data Analysis (EDA)**: Understand dataset properties, including shape, column types, and missing values.
4. **Data Preprocessing**: Clean and transform news headlines into structured text data, handling missing values and outliers.
5. **Feature Extraction**: Convert text data into numerical vectors using CountVectorizer for model training.
6. **Train-Test Split**: Split the dataset into training and testing sets to evaluate model performance.
7. **Model Training**: Train a Random Forest Classifier to predict sentiment labels.
8. **Prediction & Evaluation**: Evaluate the model using accuracy, confusion matrix, and classification report.
9. **Model Interpretation**: Analyze feature importance and insights derived from the classifier's predictions.

## Installation
To run the project locally, install the required dependencies:

```bash
pip install pandas numpy scikit-learn
```

You can also use Jupyter Notebook or Google Colab for execution.

## Usage
1. Clone the repository:
```bash
git clone <repository_url>
```
2. Navigate to the project folder:
```bash
cd Financial_Market_Analysis
```
3. Run the Jupyter Notebook (`Financial_Market_Analysis.ipynb`) in Google Colab or locally to train and evaluate the model.
4. Modify and experiment with different machine learning models for better accuracy.

## Model Performance
The model's performance is evaluated using:
- **Confusion Matrix**: Provides insights into true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Displays precision, recall, F1-score, and accuracy for each class.
- **Accuracy Score**: Measures overall performance of the classification model.

## Future Improvements
- Experiment with advanced NLP models like LSTM, Transformers, or BERT for better accuracy.
- Expand the dataset for better generalization and robustness.
- Implement real-time news sentiment analysis using web scraping and APIs.
- Improve feature engineering techniques such as TF-IDF and word embeddings.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests with enhancements, bug fixes, or new features.

