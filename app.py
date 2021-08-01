from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("grad_admissions.pkl", "rb"))



@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        gre_scr = request.form["GRE_Score"]
        toefl_scr = request.form["TOEFL_Score"]
        uni_rating = request.form["University_Rating"]
        sop = request.form["SOP"]
        lor = request.form["LOR"]
        cgpa = request.form["CGPA"]
        research = request.form['Research']

        prediction = model.predict([[gre_scr, toefl_scr, uni_rating, sop, lor, cgpa, research]])

        output = round(prediction[0], 2)

        return render_template('home.html', prediction_text="Your chance of admission is. {}".format(output))
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
