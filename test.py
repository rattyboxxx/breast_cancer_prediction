import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
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

    try:
        model = pickle.load(open("model.pkl", "rb"))
    except:
        dataset_url = "https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/master/breast-cancer-data.csv"
        df = pd.read_csv(dataset_url)

        df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

        train, test = train_test_split(df, test_size = 0.2)

        features = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']

        train_X = train[features]
        train_y = train.diagnosis

        model=RandomForestClassifier(n_estimators=100, n_jobs=-1)
        model.fit(train_X,train_y)

        with open("model.pkl", "wb") as file:
            pickle.dump(model, file)

        # Test the model if you want, acc ~ 92%
        # test_X = test[features]
        # test_y = test.diagnosis
        # prediction = model.predict(test_X)
        # acc = metrics.accuracy_score(prediction, test_y)

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

    # create the output for frontend
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
    app.run(debug=False)
