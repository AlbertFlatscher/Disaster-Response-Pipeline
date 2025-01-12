# Disaster-Response-Pipeline
Building a ML Pipeline including NLP capabilities and a Classifier able to categorize short text responses.

### Table of Contents
 
1. [Project Motivation](#motivation)
2. [Data](#data)
3. [Files](#files)
4. [Libraries](#libraries)
5. [Results](#results)
6. [Licensing](#licensing)
7. [Code Execution](#execution)

## Project Motivation <a name="motivation"></a>

Building a Disaster Response ML Pipeline is one of the projects within the Udacity Data Scientist Nanodegree Program. The training data was provided by [Appen](https://www.appen.com/)

## Data <a name="data"></a>
The data provided consist of two .csv Files:
<ul>
  <li>messages.csv - contains 26248 destinct Text messages send in disaster situations
  <li>categories.csv - maps the messages into one of 36 categories
</ul>

The categories are the following:
       'related', 'request', 'offer', 'aid_related', 'medical_help',
       'medical_products', 'search_and_rescue', 'security', 'military',
       'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report'

## Files <a name="files"></a>

<p>The following files are provided within this project:</p>

<ul>
  <li><b>ETL Pipeline Preparation.ipynb:</b> ETL Pipeline. Loads in the .csv files, cleans the data, and stores it in "DisasterResponse.db" under the table name "Responses".</li>
  <li><b>ML Pipeline Preparation.ipynb:</b> Jupyter notebook for machine learning pipeline preparation.</li>
  <li><b>process_data.py:</b> Python script refactored from "ETL Pipeline Preparation.ipynb"</li>
  <li><b>train_classifier:</b> Python script refactored from "ML Pipeline Preparation.ipynb"</li>
  <li><b>README.md:</b> This file.</li>
</ul>

<ul>
  <li><b>/WebApp:</b> File structure:</li>


<div style="font-family: 'Courier New', Courier, monospace; white-space: pre;">

- app<br>
&nbsp;&nbsp;|- template<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- master.html  # main page of web app<br>
&nbsp;&nbsp;&nbsp;&nbsp;|- go.html  # classification result page of web app<br>
&nbsp;&nbsp;|- run.py  # Flask file that runs app<br><br>

- data<br>
&nbsp;&nbsp;|- disaster_categories.csv  # data to process<br>
&nbsp;&nbsp;|- disaster_messages.csv  # data to process<br>
&nbsp;&nbsp;|- process_data.py<br>
&nbsp;&nbsp;|- InsertDatabaseName.db  # database to save clean data to<br><br>

- models<br>
&nbsp;&nbsp;|- train_classifier.py<br>
&nbsp;&nbsp;|- classifier.pkl  # saved model <br><br>

- README.md
</div>
</ul>

## Libraries <a name="libraries"></a>

I used a Jupyter notebook (Python) for the analysis. The ipynb file should run on any Python 3 version (3.0.* or higher) without any problems.</br>

Here are the additional Python libraries used within this project:

<ul>
  <li>NumPy</li>
  <li>Pandas</li>
  <li>pickle</li>
  <li>statistics</li>
  <li>SQLAlchemy</li>
  <li>NLTK (Natural Language Toolkit)</li>
  <li>Scikit-learn
    <ul>
      <li>Pipeline</li>
      <li>CountVectorizer</li>
      <li>TfidfTransformer</li>
      <li>train_test_split</li>
      <li>MultiOutputClassifier</li>
      <li>RandomForestClassifier</li>
      <li>AdaBoostClassifier</li>
      <li>FeatureUnion</li>
      <li>f1_score</li>
      <li>recall_score</li>
      <li>precision_score</li>
      <li>classification_report</li>
    </ul>
  </li>
  <li>custom_transformer (StartingVerbExtractor)</li>
</ul>

## Results <a name="results"></a>

The best performing ML model conained a CountVectorizer, StartingVerbExtractor and a MultiOutputClassifier using AdaBoost as an estimator. The best parameters for the Estimator were: n_estimators=100, learning_rate=0.01.

This enabled us to get the following metrics:
<li>Accuracy: 0.944</li>
<li>Precision: 0.932</li>
<li>Recall: 0.944</li>
<li>F1-score: 0.9301</li>

We were able to to use this pipeline for creating a Web Application which can be used for disaster response coordination.

## Licensing <a name="licensing"></a>

Thanks to [Appen](https://www.appen.com/) for providing the data.

<div class="container">
    <h2>Code Execution <a name="execution"></a></h2>

    # To create a processed sqlite db
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

    # To train and save a pkl model
    python train_classifier.py ../data/DisasterResponse.db classifier.pkl

    # To deploy the application locally
    python run.py
</div>
