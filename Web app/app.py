import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
models = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # prediction = models.predict(final_features)
    prediction = np.mean([model.predict(final_features)
                         for model in models], axis=0)

    output = prediction[0].round()
    if output == 1:
        output = "Confused"
    else:
        output = "Not confused"

    return render_template('index.html', prediction_text='Is the student confused ?: the student is {}!'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
