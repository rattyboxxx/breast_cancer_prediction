import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")


@app.route("/")
def loadPage():
    return render_template("test.html", query="")


@app.route("/", methods=["POST"])
def cancerPrediction():
    # get input from frontend
    inputQuery1 = request.form["query1"]
    inputQuery2 = request.form["query2"]
    inputQuery3 = request.form["query3"]
    inputQuery4 = request.form["query4"]
    inputQuery5 = request.form["query5"]

    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
    # print('data is: ')
    # print(data)
    # >>> 22.53, 102.1, 0.09947, 0.2225, 0.2041

    model = pickle.load(open("model.pkl", "rb"))

    # Create the pandas DataFrame
    new_df = pd.DataFrame(
        data,
        columns=[
            "texture_mean",
            "perimeter_mean",
            "smoothness_mean",
            "compactness_mean",
            "symmetry_mean",
        ],
    )

    # calculate prob of input
    single = model.predict(new_df)
    probability = model.predict_proba(new_df)[:, 1][0]
    # print(probability)

    if single == 1:
        output = "The patient is diagnosed with Breast Cancer"
        output1 = "Confidence: {}%".format(probability * 100)
    else:
        output = "The patient is not diagnosed with Breast Cancer"
        output1 = "Congrats!"

    return render_template(
        "test.html",
        output1=output,
        output2=output1,
        query1=request.form["query1"],
        query2=request.form["query2"],
        query3=request.form["query3"],
        query4=request.form["query4"],
        query5=request.form["query5"],
    )


if __name__ == "__main__":
    app.run()
