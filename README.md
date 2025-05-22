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
