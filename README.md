# question-answering-with-bert
Fine-Tuning BERT on SQuAD v1.1 for Question Answering
This notebook demonstrates the complete pipeline for fine-tuning a pre-trained BERT model (bert-base-uncased) on a custom CSV version of the SQuAD v1.1 dataset, using the Hugging Face transformers library and TensorFlow/PyTorch backend.

Overview
The notebook covers every step of the fine-tuning process:

Data Preparation

Load a CSV version of the SQuAD v1.1 dataset.

Split the dataset into training, validation, and test sets (80/10/10).

Convert the data into the datasets library format suitable for transformers.

Data Preprocessing

Use the BertTokenizerFast to tokenize context-question pairs.

Map character-based answer spans into token positions (start and end).

Discard unnecessary offset mappings after position alignment.

Model Setup

Load the pre-trained bert-base-uncased model with a Question Answering head (BertForQuestionAnswering).

Define TrainingArguments such as batch size, learning rate, and evaluation strategy.

Training

Train the model using Hugging Face's Trainer API.

Evaluate the model during training using validation data.

Technologies & Libraries
Python 3.x

Hugging Face Transformers

Datasets Library

pandas, NumPy, scikit-learn

TensorFlow / PyTorch backend (Trainer-compatible)
File Structure
dataset/SQuAD-v1.1.csv — Raw dataset in CSV format

dataset/train.csv, test.csv — Split subsets for training and evaluation

dataset/results/ — Output directory for saving checkpoints and logs

Use Case
This notebook is ideal for anyone looking to:

Fine-tune a transformer model on a custom QA dataset

Learn the tokenization and preprocessing nuances of QA tasks

Reproduce QA model training with Hugging Face's high-level APIs

