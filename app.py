from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import sys
import pandas as pd
import numpy as np
from src.logger import logger
from src.exception import ZomatoDeliveryException
from src.configuration.configuration import *
from src.constants import *
from src.pipeline.prediction_pipeline import Prediction_Pipeline, CustomData
from src.pipeline.training_pipeline import Train
from src.pipeline.batch import Batch_prediction, PREDICTION_FOLDER

ALLOWED_EXTENSIONS = {'csv'}
UPLOAD_FOLDER = os.path.join(os.getcwd(), PREDICTION_FOLDER, 'uploaded_csv')

# Model paths
feature_eng_file_path = FEATURE_ENG_MODEL_PATH
transformer_file_path = PROCESSOR_MODEL_PATH
model_file_path = MODEL_FILE_PATH

app = Flask(__name__, template_folder='templates')


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    try:
        logger.info("Received request for single data prediction.")
        data = CustomData(
            Delivery_person_Age=int(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings=float(request.form.get('Delivery_person_Ratings')),
            Weather_conditions=request.form.get('Weather_conditions'),
            Road_traffic_density=request.form.get('Road_traffic_density'),
            Vehicle_condition=int(request.form.get('Vehicle_condition')),
            multiple_deliveries=int(request.form.get('multiple_deliveries')),
            distance=float(request.form.get('distance')),
            Type_of_order=request.form.get('Type_of_order'),
            Type_of_vehicle=request.form.get('Type_of_vehicle'),
            Festival=request.form.get('Festival'),
            City=request.form.get('City'),
            Delivery_person_City=request.form.get('Delivery_person_City')
        )

        final_data = data.get_data_as_dataframe()
        predict_pipeline = Prediction_Pipeline()
        prediction = predict_pipeline.predict(final_data)

        result = round(prediction[0], 2)
        logger.info(f"Prediction result: {result}")
        return render_template('form.html', final_result=result)

    except Exception as e:
        logger.exception("Error during single prediction")
        return render_template('form.html', error="Invalid input or server error. Please check values.")


@app.route('/batch', methods=['GET', 'POST'])
def perform_batch_prediction():
    if request.method == 'GET':
        return render_template('batch.html')
    
    try:
        file = request.files.get('csv_file')

        if not file or '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
            logger.warning("Invalid file upload attempt.")
            return render_template("batch.html", prediction_type='batch', error="Invalid file type")

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Clear previous uploaded files
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Save new file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        logger.info(f"CSV uploaded for batch prediction: {filename}")

        # Perform batch prediction
        batch = Batch_prediction(file_path, model_file_path, transformer_file_path, feature_eng_file_path)
        batch.start_batch_prediction()

        logger.info("Batch prediction completed.")
        return render_template("batch.html", prediction_result="Batch Prediction Done", prediction_type='batch')

    except Exception as e:
        logger.exception("Error during batch prediction")
        return render_template("batch.html", prediction_type='batch', error="Batch prediction failed.")


@app.route('/train', methods=['GET', 'POST'])
def training():
    if request.method == 'GET':
        return render_template('train.html')
    
    try:
        logger.info("Starting training pipeline...")
        pipeline = Train()
        pipeline.run_pipeline()

        logger.info("Model training completed successfully.")
        return render_template("train.html", message='Training completed')

    except Exception as e:
        logger.exception("An error occurred during training.")
        return render_template("train.html", error="Training failed. Check logs for more info.")


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8888)
