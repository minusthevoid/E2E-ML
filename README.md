# E2E-ML
ML Pipeline 

This is my attempt to create an end to end pipeline for image classification. This includes
- Data Ingestion
- Data Processing
- Data Splitting
- Model Training
- Model Evaluation

## Installation

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

Will attempt
- model deployment
- monitoring model performance.

## Installation

Install project dependencies with:

## Installation

Install the dependencies using:

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

## Running the Pipeline

You can run the full pipeline using:

```bash
python run_pipeline.py --num 10 --dir data
```
To use different categories:
```bash
python run_pipeline.py --search "birds,cars" --num 10 --dir data
```

## Configuration

Default classification categories are specified in `e2e_ml/config.py` as
`CLASS_LABELS`. Update this list to change the images downloaded and processed.
You can also override the categories at runtime via the `--search` option.

## Running Tests

Install the project dependencies and run tests with `pytest`:

```bash
pytest
```
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


After installing the dependencies you can run the tests with `pytest`:

Install the project dependencies and run tests with `pytest`:


```bash
pytest
```