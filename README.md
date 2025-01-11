# Disaster-Response-Pipeline
Building a ML Pipeline including NLP capabilities and a Classifier able to categorize short text responses.

### Table of Contents
 
1. [Project Motivation](#motivation)
2. [Data](#data)
3. [Files](#files)
4. [Libraries](#libraries)
5. [Results](#results)
6. [Licensing](#licensing)

## Project Motivation <a name="motivation"></a>

Building a Disaster Response ML Pipeline is one of the projects within the Udacity Data Scientist Nanodegree Program. The training data was provided by [Appen](https://www.appen.com/)

## Data <a name="data"></a>
The data provided consist of three .csv Files:
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

The following files are provided within this project:

<ul>
  <li><b>DisasterResponse.db<b>
  <li><b>ETL Pipeline Preparation.ipynb<b>
  <li><b>ML Pipeline Preparation.ipynb<b>
  <li><b>process_data.py<b>
  <li><b>README.md<b> This file</li>
</ul>

## Libraries <a name="libraries"></a>

I used a Jupyter notebook (Python) for the analysis. The ipynb file should run on any Python 3 version (3.0.* or higher) without any problems.</br>

Here are the additional Python libraries used within this project:

<ul>
  <li>Numpy Version "1.26.4"</li>
  <li>Pandas Version "2.2.2"</li>
  <li>Matplotlib Version "3.8.4"</li>
  <li>seaborn Version "0.13.2"</li>
  <li>glob Version "0.5.0"</li>
</ul>

You will need to download Stackoverflow’s Annual Developer Surveys from 2019 till 2024 and put them in a path structure of this form:

os.path.join('..', 'data', 'stack-overflow-developer-survey-' + year)

[Here](https://insights.stackoverflow.com/survey) you can find the data. </br>

## Results <a name="results"></a>

The aim of the analysis was to get an overview of the most used programming languages from 2019 till 2024. From this starting point it was tried to find trends and conclusions.

The results can be found on Medium [Blog](https://medium.com/p/f1930bfe91bc/edit).

## Licensing <a name="licensing"></a>

Thanks to Stack Overflow for providing the data.
