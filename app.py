import flask
import model

app = flask.Flask(__name__)

@app.route('/')
def home():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if (flask.request.method == 'POST'):
        features = [x for x in flask.request.form.values()]

        sorted_output = model.predict(features[0])

        res = flask.render_template('index.html', prediction_text='Product Recommendation for user : {}'.format(features[0]),
                                    tables=[sorted_output.to_html(classes='data')], titles=sorted_output.columns.values)
        return res
    else:
        return flask.render_template('indexgi.html')

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
