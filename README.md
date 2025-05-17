# Veracity Vigilance: Fake News Detection

This project aims to detect fake news articles using machine learning techniques. It leverages TF-IDF vectorization for text preprocessing and Logistic Regression as the classifier. The system also features a Streamlit web app for easy interaction, confidence scoring for predictions, and explainability using the ELI5 library.

## Features
- Text preprocessing using TF-IDF vectorizer
- Logistic Regression classifier for fake news detection
- Streamlit-based interactive web application
- Confidence scoring to indicate prediction certainty
- Model explainability using ELI5 for transparency

## Project Structure
- `app.py` - Main Streamlit app file to run the user interface
- `model.pkl` - Serialized Logistic Regression model
- `tfidf.pkl` - Serialized TF-IDF vectorizer
- `notebook.ipynb` - Jupyter notebook used for training and evaluation
- `requirements.txt` - List of Python dependencies

## Getting Started

### Prerequisites
Make sure you have Python 3.7+ installed.

### Installation

Run these commands in your terminal to set up the project:

```bash
git clone https://github.com/Mahesh-Hemadri/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
