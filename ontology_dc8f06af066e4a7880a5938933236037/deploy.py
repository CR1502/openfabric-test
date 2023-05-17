from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load the h5 model
model = tf.keras.models.load_model('D:\Coding\Python\PROJECTS\openfabric-test\openfabric-test\ontology_dc8f06af066e4a7880a5938933236037\chatbot_model.h5')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['input_text']

        # Make predictions using the loaded model
        prediction = model.predict([input_text])

        # Return the prediction as a JSON response
        response = {'prediction': prediction}
        return jsonify(response)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run()
