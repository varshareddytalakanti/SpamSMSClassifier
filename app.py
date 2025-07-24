from flask import Flask, request, render_template
import pickle
import os
app = Flask(__name__)
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = vectorizer.transform([message])
    prediction = model.predict(data)
    return render_template('index.html', result="Spam" if prediction[0] else "Not Spam")

if __name__ == '__main__':

 import os
app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 3000)))



