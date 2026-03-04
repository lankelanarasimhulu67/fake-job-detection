## Deployment Details

This project **“Fake Recruitment Detection in Online Portals”** is a Machine Learning web application that detects whether a job posting is fake or legitimate using Natural Language Processing and classification algorithms.

### Deployment Platform

The application is deployed using **Render**, a cloud platform that allows hosting web applications directly from a GitHub repository.

### Repository Link

Source code is available on GitHub:
https://github.com/lankelanarasimhulu67/fake-job-detection

### Live Application

The deployed web application can be accessed here:
https://fake-job-detection-2.onrender.com

### Technologies Used

* Python
* Flask (Web Framework)
* Scikit-learn (Machine Learning Model)
* Pandas and NumPy (Data Processing)
* Gunicorn (Production Web Server)

### Important Project Files

The project contains the following main files and folders:

* **app.py** – Main Flask application that handles user requests and predictions
* **train.py** – Script used to train the machine learning model
* **model.pkl** – Saved trained model used for prediction
* **requirements.txt** – List of Python libraries required to run the project
* **templates/** – Contains HTML files for the web interface
* **static/** – Contains CSS or frontend resources
* **Procfile** – Defines the start command for deployment on Render

### How Deployment Works

1. The project code is stored in a GitHub repository.
2. Render connects to the GitHub repository.
3. Render installs required dependencies using `requirements.txt`.
4. The application is started using Gunicorn with the Flask app.
5. Render provides a public URL where the application can be accessed online.

### Outcome

The system allows users to enter a job description and automatically predicts whether the job posting is **Fake** or **Real** using the trained machine learning model.
