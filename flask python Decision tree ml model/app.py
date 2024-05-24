 # Import necessary modules
from flask import Flask, request, render_template
import pickle
import numpy as np

# Create a Flask application instance
app = Flask(__name__, template_folder='.')

# load the trained model 
with open("decision_tree_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define a route for the homepage
@app.route("/")
def home():
    return render_template("index.html") # Render the index.html template

# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    # get input data from the form
    age = float(request.form["age"]) # Extract age input from the form and convert to float
    gender = float(request.form["gender"]) # Extract gender input from the form and convert to float

    # make a prediction using the model
    input_data = [[age, gender]] # Combine age and gender into a list
    prediction = model.predict(input_data) # Make a prediction using the model

    # pass the prediction value to the template
    return render_template("index.html", prediction=prediction[0]) # Render the index.html template with prediction value
# Start the Flask application
if __name__ == "__main__":
    app.run(debug=True) # Run the application in debug mode for development



