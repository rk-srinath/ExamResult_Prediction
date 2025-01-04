import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pre-trained Random Forest model
with open('exam_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    # Render the index.html for user input
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get inputs from the form
    study_hours = request.form.get('study_hours')
    exam_score = request.form.get('exam_score')

    try:
        # Convert inputs to appropriate numeric format
        input_data = [[float(study_hours), float(exam_score)]]

        # Make a prediction
        prediction = model.predict(input_data)[0]

        # Map the prediction to a user-friendly output
        result = "Pass" if prediction == 1 else "Fail"
    except Exception as e:
        result = f"Error in prediction: {e}"

    # Render the result.html with the prediction
    return render_template('result.html', prediction=f"The predicted result is: {result}")

if __name__ == '__main__':
    app.run(debug=True)
