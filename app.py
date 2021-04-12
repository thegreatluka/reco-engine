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

if __name__ == '__main__':
    app.debug=True
    app.run()
