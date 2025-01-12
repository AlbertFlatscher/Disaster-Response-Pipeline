import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from custom_transformer import StartingVerbExtractor


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
try:
    print("Connecting to database...")
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('Responses', engine)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")

# load model
try:
    print("Loading model...")
    model = joblib.load("../models/classifier.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    try:
        print("Rendering index page...")
        genre_counts = df.groupby('genre').count()['message']
        genre_names = list(genre_counts.index)
        print(f"Genres found: {genre_names}")
    
        # create visuals
        # TODO: Below is an example - modify to create your own visuals
        graphs = [
            {
                'data': [
                    Bar(
                        x=genre_names,
                        y=genre_counts
                    )
                ],

                'layout': {
                    'title': 'Distribution of Message Genres',
                    'yaxis': {
                        'title': "Count"
                    },
                    'xaxis': {
                        'title': "Genre"
                    }
                }
            }
        ]
    
        # encode plotly graphs in JSON
        ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
        graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
        print("Index page rendered successfully.")
        return render_template('master.html', ids=ids, graphJSON=graphJSON)
    except Exception as e:
        print(f"Error rendering index page: {e}")
        return str(e)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    try:
        print("Rendering go page...")
        query = request.args.get('query', '') 
        print(f"Received query: {query}")

        # use model to predict classification for query
        classification_labels = model.predict([query])[0]
        print(f"Predicted classifications: {classification_labels}")

        classification_results = dict(zip(df.columns[4:], classification_labels))
        print("Go page rendered successfully.")
        # This will render the go.html Please see that file. 
        return render_template(
            'go.html',
            query=query,
            classification_result=classification_results
        )
    except Exception as e:
        print(f"Error rendering go page: {e}")
        return str(e)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()