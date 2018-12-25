# Disaster Response Pipeline Project

This is the 5th project for Udacity's Datascience Nanodegree program. We attempt to classify messages sent during a disaster scenario, in order to know which services should be sent to which locations.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Explanation of the Files:
- `data/process_data.py`: The ETL pipeline used to process data and save it in order to build our model.
- `models/train_classifier.py`: The Machine Learning pipeline used to fit, optimize, evaluate, and export the model to a Python pickle file (the pickle file isn't added to the repo due to size limitations).
- `app/templates/*.html`: HTML templates for the web app.
- `run.py`: Starts the server for the web app and prepares the 3 visualizations.
