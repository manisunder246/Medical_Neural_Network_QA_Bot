# Neural Network-Based Medical Question Answering System

This repository contains the complete code and resources for a Neural Network-based Question Answering (NNQA) system developed as part of an M.Tech AIML coursework assignment. The objective was to design and implement a QA system capable of answering medical-related questions using deep learning models.

## Project Overview

### Problem Statement:
The task involves developing a question-answering system using a neural network that leverages fine-tuned models to handle medical-related questions. The NNQA system is capable of fetching answers from domain-specific data sources like Wikipedia, Healthline, and medical research articles.

### Features:
- Fine-tuned **RoBERTa** model for the medical domain.
- RESTful APIs for backend communication using **Flask**.
- A simple, user-friendly frontend built with **HTML**, **CSS**, and **JavaScript**.
- **Multilingual** support with potential for further enhancements.
- Preprocessing of domain-specific datasets from Wikipedia and Healthline using **BeautifulSoup**, **Selenium**, and NLP techniques like **Spacy** and **NLTK**.

## How to Run the Application Locally

### Prerequisites:
- **Python 3.7+**
- Install all dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt

## Files in the Repository:

- `fine_tuned_roberta/`: Folder containing the fine-tuned RoBERTa model.
- `main_code.ipynb`: Python code for model development, including training and evaluation.
- `QA_bot.ipynb`: Code for Flask deployment of the QA bot.
- `index.html`: Frontend HTML code for the user interface.
- `qabot.py`: Flask application handling the API requests for the bot.
- `background.jpg`: Background image used for the bot UI.
- `requirements.txt`: Required dependencies for running the application.

## Dataset & Model

The QA bot was trained using medical-related datasets from:
- **Wikipedia articles**: Articles on various medical topics.
- **Healthline blog articles**: Articles related to drugs and medical conditions.

The model used:
- **RoBERTa (deepset/roberta-base-squad2)** fine-tuned for question-answering in the medical domain.

## Future Enhancements
- **Multilingual support**: For cross-lingual question-answering capabilities.
- **Knowledge graph integration**: To enhance reasoning and provide structured knowledge.
- **Multi-hop reasoning**: To allow the model to answer more complex queries.
- **Ensemble models**: Combining models like BERT, RoBERTa, and BioBERT for more robust performance.

