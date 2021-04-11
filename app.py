import flask
import numpy as np
import pandas as pd

app = flask.Flask(__name__)
reco_df_load = pd.read_pickle(r"https://github.com/thegreatluka/reco-engine/blob/master/pikles/user_final_rating_df.pkl?raw=true")
sent_df_load = pd.read_pickle(r"https://github.com/thegreatluka/reco-engine/blob/master/pikles/items_by_sentiment_score.pkl?raw=true")

@app.route('/')
def home():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if (flask.request.method == 'POST'):
        int_features = [x for x in flask.request.form.values()]
        final_features = [np.array(int_features)]
        # output = model_load.predict(final_features).tolist()
        output = reco_df_load.loc[int_features[0]].sort_values(ascending=False)[0:20]
        sorted_output = sent_df_load.loc[output.index.tolist()].sort_values(by='Pos%', ascending=False)[0:5]
        res = flask.render_template('index.html', prediction_text='Product Recommendation for : {}'.format(int_features[0]),
                                    tables=[sorted_output.to_html(classes='data')], titles=sorted_output.columns.values)
        return res
    else:
        return flask.render_template('index.html')

@app.route("/predict_api", methods=['POST', 'GET'])
def predict_api():
    print(" request.method :",flask.request.method)
    if (flask.request.method == 'POST'):
        data = flask.request.get_json()
        # res = flask.jsonify(model_load.predict([np.array(list(data.values()))]).tolist())
        res = "Result2"
        return res
    else:
        return flask.render_template('index.html')

if __name__ == '__main__':
    app.debug=True
    app.run()
