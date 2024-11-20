# models.py

import pickle
import logging

def load_pipelines():
    pipelines = {}
    model_files = ['xgb_pipeline.pkl', 'rf_pipeline.pkl', 'knn_pipeline.pkl']
    for filename in model_files:
        try:
            with open(filename, 'rb') as file:
                pipeline = pickle.load(file)
                model_name = filename.replace('_pipeline.pkl', '').upper()
                pipelines[model_name] = pipeline
        except FileNotFoundError:
            logging.error(f"Model file {filename} not found.")
    return pipelines
