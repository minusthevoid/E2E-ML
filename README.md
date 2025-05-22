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

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

You can run the full pipeline using:

```bash
python run_pipeline.py --search "cats,dogs" --num 10 --dir data
```

## Running Detection

After the model is trained you can detect cats and dogs in a folder of images:

```bash
python run_detector.py --input data/cats --output detections
```

## Running Tests

Install the project dependencies and run tests with `pytest`:

```bash
pytest
```
