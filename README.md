# Sentiment Analysis App using Streamlit

This repository contains a simple web application built with Streamlit for sentiment analysis of restaurant reviews. The application uses a trained Support Vector Machine (SVM) model and a bag-of-words approach for text classification.

## Getting Started

To run the app locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required packages by running: `pip install -r requirements.txt`
3. Download the `Restaurant_Reviews.tsv` dataset and place it in the same directory as `app.py`.
4. Ensure you have the `classifier_model.pkl` and `count_vectorizer.pkl` files available in the same directory.
5. Run the Streamlit app: `streamlit run app.py`

## Usage

Once the app is running, you can interact with it in your web browser. Enter a restaurant review text in the provided text area and click the "Predict" button to see the predicted sentiment label (Positive or Negative).

## Model Metrics

The app also displays model evaluation metrics such as Accuracy, Recall, and F1 Score based on the confusion matrix of the SVM model. Additionally, K-Fold Cross Validation mean and standard deviation are provided.

## Files and Structure

- `app.py`: The Streamlit app code for user interaction and prediction.
- `classifier_model.pkl`: The trained SVM model for sentiment prediction.
- `count_vectorizer.pkl`: The saved CountVectorizer used to preprocess text data.
- `Restaurant_Reviews.tsv`: The dataset containing restaurant reviews for training.
- `requirements.txt`: Required Python packages for running the app.
- `README.md`: This file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute, open issues, or provide feedback!
