from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model/crop_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        prediction = model.predict(
            [[N, P, K, temperature, humidity, ph, rainfall]]
        )

        result = prediction[0]

        return render_template(
            'index.html',
            prediction_text=f"Recommended Crop: {result}"
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == '__main__':
    app.run(debug=True)