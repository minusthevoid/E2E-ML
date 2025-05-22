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

## Running the Pipeline

You can run the full pipeline using:

```bash
python run_pipeline.py --search "cats,dogs" --num 10 --dir data
```

## Running Tests

After installing the dependencies you can run the tests with `pytest`:

```bash
pytest
```
