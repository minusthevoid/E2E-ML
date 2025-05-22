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

Install project dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

You can run the preprocessing and augmentation pipeline using:

```bash
python run_pipeline.py --search "cats,dogs" --num 10 --dir data
```

## Full Automation

To download images, process them and train the classifier in one step run:

```bash
python run_all.py --search "cats,dogs" --num 10 --dir data
```

## Running Tests

Install the project dependencies and run tests with `pytest`:

```bash
pytest
```
