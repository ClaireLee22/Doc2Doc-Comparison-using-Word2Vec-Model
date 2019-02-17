# Doc2Doc-Comparison-using-Word2Vec-Model
Deep Learning Project [UCSC Silicon Valley Extension: Deep Learning and Artificial Intelligence with TensorFlow]
- Cowork with Santosh Honnavalli

## Project Overview
### Project Description
Identify the similarity among two thousands of documents based on the Word2Vec skip-gram model.

### Project Procedure
- Prepare the data
  - Collect 40000 documents
- Preprocesse the data
  - Remove non-English words
  - Remove stopwords
  - Tokenization
  (Data_preparion1.py) 
  (Data_preparion2.py)
- Build Word2Vec model
- Train the  model
- Doc2Doc comparison
  (Doc2Doc_comparison.py)
- Verify results by exact word match count
  (match_word_count.py)
### Project Results
- Word2Vec Visualization
- Loss graph
- Comparison output(show 5 documents as an example)
- word_match_count output (verify Word2Vec model accuracy)

## Getting Started
### Prerequisites

This project requires **Python 3.6** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [tensorflow](https://www.tensorflow.org/install/pip)

### Run

In a terminal or command window, run one of the following commands:

```bash
python Doc2Doc_comparison.py
```  

### Data

4000 healthcare related documents are using a private dataset
