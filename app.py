from flask import Flask, request, render_template
import numpy as np
import pickle

# Load pre-trained model and scalers
model = pickle.load(open(r'C:\Users\MD MOKARRAM ALAM\PycharmProjects\PythonProject1/model/model.pkl', 'rb'))
standard_scaler = pickle.load(open(r'C:\Users\MD MOKARRAM ALAM\PycharmProjects\PythonProject1/model/standscaler.pkl', 'rb'))
minmax_scaler =  pickle.load(open(r'C:\Users\MD MOKARRAM ALAM\PycharmProjects\PythonProject1/model/minmaxscaler.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

# Crop mapping dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
    10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommendation")
def recommendation():
    return render_template("recommend.html", result=None)  # Pass None initially

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form data
        nitrogen = float(request.form.get("Nitrogen", 0))
        phosphorus = float(request.form.get("Phosphorus", 0))
        potassium = float(request.form.get("Potassium", 0))
        temperature = float(request.form.get("Temperature", 0))
        humidity = float(request.form.get("Humidity", 0))
        ph = float(request.form.get("Ph", 0))
        rainfall = float(request.form.get("Rainfall", 0))

        # Prepare input data
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        # Apply preprocessing
        scaled_features = minmax_scaler.transform(input_data)  # Apply MinMaxScaler
        final_features = standard_scaler.transform(scaled_features)  # Apply StandardScaler

        # Make prediction
        prediction = model.predict(final_features)[0]

        # Map prediction to crop name
        crop = crop_dict.get(prediction, "Unknown Crop")
        result = f"Crop Recommended: {crop}"

        # Return result
        return render_template("recommend.html", result=result)

    except Exception as e:
        return render_template("recommend.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
