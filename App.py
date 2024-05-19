from flask import flask, request, render_template
import pickle

app = flask(__name__)

model = pickle.load(open("../model.pkl", "rb")) #opens the file in read mode
tfidf = pickle.load(open("../tfidf.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    
    text = request.form.get('Revieww')
    corpus = []
    corpus.append(text)
    corpus = tfidf.transform(corpus)
    pred = model.predict(corpus)

    return render_template("index.html", pred_text=pred)

if __name__ == "__main__":
    app.run(debug=True)