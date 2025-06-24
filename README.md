# 🍽️ Zomato Delivery Time Prediction

<div align="center">

[![GitHub](https://img.shields.io/github/stars/ArpitKadam/Zomato-Delivery-Time-Prediction?style=social)](https://github.com/ArpitKadam/Zomato-Delivery-Time-Prediction)
[![GitHub issues](https://img.shields.io/github/issues/ArpitKadam/Zomato-Delivery-Time-Prediction)](https://github.com/ArpitKadam/Zomato-Delivery-Time-Prediction/issues)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-Enabled-orange.svg)
![Dagshub](https://img.shields.io/badge/Dagshub-Enabled-brightgreen.svg)

[License](https://github.com/ArpitKadam/Zomato-Delivery-Time-Prediction/blob/main/LICENSE) | [Dagshub](https://dagshub.com/ArpitKadam/Zomato-Delivery-Time-Prediction) 

</div>

---

## 📋 Overview

This project predicts food delivery time using machine learning techniques based on customer location, restaurant distance, weather, and order details. It employs a full ML pipeline including DVC for data versioning, MLflow for experiment tracking, Docker for containerization, and deployment-ready Flask endpoints.

---

## 📑 Table of Contents

- 🚀 Installation
- 🛠️ Environment Setup
- 📊 Dagshub & MLflow Setup
- 💾 DVC Pipeline
- 🤖 Model Training and Prediction
- 🐳 Docker Deployment

---

## 📁 Project Structure

<details>
<summary>Click to expand/collapse</summary>
  
```
Zomato-Delivery-Time-Prediction/
├── README.md
├── LICENSE
├── Dockerfile
├── .dockerignore
├── .dvcignore
├── app.py
├── main.py
├── requirements.txt
├── setup.py
├── template.py
├── init.py
│
├── Artifacts/
│ ├── Data_Ingestion/
│ ├── Data_Transformation/
│ └── Model_Training/
│
├── batch_prediction/
│ ├── raw_input/
│ ├── feature_eng/
│ ├── transformed/
│ ├── prediction_csv/
│ └── uploaded_csv/
│
├── Research/
│ └── research.ipynb
│
├── templates/
│ └── *.html
│
├── src/
│ ├── components/
│ ├── configuration/
│ ├── constants/
│ ├── exception/
│ ├── logger/
│ ├── pipeline/
│ └── utils/
```
</details>

---

## 🚀 Installation

```bash
git clone https://github.com/ArpitKadam/Zomato-Delivery-Time-Prediction.git
cd Zomato-Delivery-Time-Prediction
```

```bash
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate (Linux/Mac)
pip install -r requirements.txt
```

📊 Dagshub & MLflow Setup
```python
# init dagshub tracking
from dagshub import dagshub_logger
dagshub_logger.init("Zomato-Delivery-Time-Prediction", "ArpitKadam", mlflow=True)

# set tracking URI for MLflow
import os
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/ArpitKadam/Zomato-Delivery-Time-Prediction.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "ArpitKadam"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<your_token>"
```

💾 DVC Pipeline
Run Stages
```
dvc repro
```

Push to Remote
```bash
dvc remote add -d dagshub https://dagshub.com/ArpitKadam/Zomato-Delivery-Time-Prediction.dvc
dvc remote modify dagshub user ArpitKadam
dvc remote modify dagshub password <your_token>
dvc push
```

🤖 Model Training & Prediction
```bash
python main.py         # for training pipeline
python src/pipeline/batch.py   # for batch prediction
python app.py          # Flask web interface
```

🐳 Docker Deployment
Build & Run Locally
```bash
docker build -t zomato-delivery-app .
docker run -p 5000:5000 zomato-delivery-app
```

Docker Hub
```bash
docker tag zomato-delivery-app <your-username>/zomato-delivery-app
docker push <your-username>/zomato-delivery-app
```

## 🤝 Contributions

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

---

## 📄 License

This project is licensed under the [GPL-3.0 license](https://github.com/ArpitKadam/Zomato-Delivery-Time-Prediction/blob/main/LICENSE).

---

## 📬 Contact

- Email: [arpitkadam922@gmail.com](mailto:arpitkadam922@gmail.com)
- GitHub: [ArpitKadam](https://github.com/ArpitKadam)
- Personal: [ArpitKadam](https://arpit-kadam.netlify.app/)

---

<div align="center">
Made with ❤️ by ArpitKadam
</div>
