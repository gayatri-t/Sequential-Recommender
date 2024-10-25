# Sequential-Recommender


This repository contains the implementation of a Sequential Recommender System using TensorFlow. The system leverages deep learning techniques, including Attention Mechanisms, Matrix Factorization (using Singular Value Decomposition), and Neural Collaborative Filtering (NCF), to provide personalized recommendations based on user interaction history.

## Project Overview

This project aims to build a recommender system that can predict user preferences and make recommendations based on their historical interactions. The model incorporates advanced techniques such as:

- **Attention Mechanisms**: To weigh the importance of different user interactions.
- **Matrix Factorization**: To reduce dimensionality and capture latent factors in user-item interactions.
- **Neural Collaborative Filtering**: To create a hybrid model that combines the strengths of collaborative filtering and deep learning.

## Dataset

The datasets used for training and testing are in TSV (Tab-Separated Values) format and contain the following columns:

- `rec_his`: User's recommendation history.
- `src_his`: User's search history.
- `ts`: Timestamps of interactions.
- `label`: Binary label indicating whether the user interacted with a recommendation (1) or not (0).

### Data Files

- `dataset/train_inter.tsv`: Training dataset.
- `dataset/test_inter.tsv`: Testing dataset.

## Installation

To run this project, make sure you have Python 3.x installed. You can install the required packages using pip:

```bash
pip install numpy pandas tensorflow scikit-learn fastFM matplotlib seaborn
