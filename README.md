# E2E-ML
ML Pipeline 

This is my attempt to create an end to end pipeline for image classification. This includes
- Data Ingestion
- Data Processing
- Data Splitting
- Model Training
- Model Evaluation

Will attempt
- model deployment
- monitoring model performance.

## Installation

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

You can run the full pipeline using:

```bash
python run_pipeline.py --search "cats,dogs" --num 10 --dir data
```

## Running Tests

Install the project dependencies and run tests with `pytest`:

```bash
pytest
```

## Training and Classification

Run the automated pipeline, train a simple model, and classify new images:

```bash
python run_classifier.py --train_num 20 --test_num 5 --dir data
```

The script downloads images of cats and dogs for training and testing, trains a
logistic regression model, and prints predictions for the test images.
