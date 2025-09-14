from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        entity = int(request.form['entity'])
        year = int(request.form['year'])

        features = np.array([[entity, year]])
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f"Predicted Solar Capacity: {output}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
