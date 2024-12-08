# MLP_Final_Project
Exploring Variants of BERT and Hyperparameter Optimization for Sentence Classification on the MRPC Dataset

Overview
This project provides the resources and implementation for analyzing different variants of the BERT model alongside hyperparameter tuning for sentence classification. Using the Microsoft Research Paraphrase Corpus (MRPC) dataset, the objective is to evaluate the impact of model variants and hyperparameter configurations on performance, offering insights into effective practices for NLP tasks.

Introduction to BERT
BERT, or Bidirectional Encoder Representations from Transformers, is a state-of-the-art technique for pre-training language models. It involves training on large corpora like Wikipedia to develop a robust understanding of language, enabling its subsequent fine-tuning for specific Natural Language Processing (NLP) applications, such as question answering or paraphrase identification.

Key Feature: Bidirectional Contextual Understanding
BERT’s defining characteristic is its deeply bidirectional approach, capturing the meaning of a word by analyzing the context provided by both preceding and following words. For example, in "I made a bank deposit," the model understands "bank" by considering the surrounding context, differentiating between meanings like a financial institution or a riverbank.

Training Framework
Core Training Tasks
Masked Language Modeling:
Predicting masked words in a sentence where 15% of words are hidden during training.
Next Sentence Prediction:
Determining whether one sentence logically follows another to enhance sentence-pair understanding.
Pre-training and Fine-tuning
The pre-training phase is resource-intensive, typically requiring days on high-performance hardware such as TPUs. However, this effort is conducted once, with the resulting pre-trained models being fine-tuned for downstream tasks (e.g., MRPC classification) using less computational resources.

Dataset Information
The Microsoft Research Paraphrase Corpus (MRPC) dataset is used, which contains pairs of sentences labeled to indicate whether they are paraphrases.

Training Set: 3,500 pairs.
Validation Set: 1,700 pairs. The dataset and preprocessing scripts are included in the data/ directory.
Project Structure
data/: Dataset and preprocessing scripts.
notebooks/: Jupyter notebooks for experimentation, results, and analysis.
scripts/: Python scripts for training, evaluation, and tuning.
results/: Performance metrics and analysis outputs.
requirements.txt: Dependencies for the project.

Setup Instructions
Clone Repository:
git clone <repository-url>
cd bert-hyperparameter-tuning

Set Up Virtual Environment:
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

Install Dependencies:
pip install -r requirements.txt

Usage Instructions
Data Preparation
Preprocess the MRPC dataset:
python scripts/preprocess_data.py

Model Training
Train the BERT model with specified hyperparameters:
python scripts/train_model.py --learning_rate 2e-5 --batch_size 16 --num_epochs 3

Evaluation
Evaluate the model:
python scripts/evaluate_model.py

Interactive Exploration
Open the Jupyter notebook for detailed exploration:
jupyter notebook notebooks/bert_experiments.ipynb

Hyperparameter Tuning
The following configurations were explored:
Learning Rate: 1e-5, 2e-5, 3e-5.
Batch Size: 8, 16, 32.
Number of Epochs: 2, 3, 4.
Results and insights are detailed in the notebook and summarized in the results/ directory.

Key Results
The experiments highlight:

BERT-Base vs. BERT-Large performance comparison.
Impact of hyperparameters (e.g., learning rate) on accuracy, precision, recall, and F1 scores.
Influence of batch size and epochs on model convergence.
The findings aim to guide optimal model and hyperparameter selection for similar NLP tasks.
