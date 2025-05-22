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
python run_pipeline.py --search "cats,dogs" --num 10 --dir data
```

## Running Tests

Install the project dependencies and run tests with `pytest`:

```bash
pytest
```

## Running Detection

After running the pipeline you can draw bounding boxes around cats and dogs using a
pretrained model. Provide the folder of images and an output folder:

```bash
python run_detector.py input_dir output_dir
```
