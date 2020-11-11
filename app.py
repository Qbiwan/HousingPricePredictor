import joblib
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model/model.h5")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def main():

    if request.method == "GET":
        return render_template('main.html')

    if request.method == "POST":

        inputs = {}
        inputs['medianIncome'] = request.form['medianIncome']
        inputs['housingMedianAge'] = request.form['housingMedianAge']
        inputs['totalRooms'] = request.form['totalRooms']
        inputs['totalBedrooms'] = request.form['totalBedrooms']
        inputs['population'] = request.form['population']
        inputs['households'] = request.form['households']
        inputs['latitude'] = request.form['latitude']
        inputs['longitude'] = request.form['longitude']

        preds = predict(
            model=model,
            scaler=scaler,
            inputs=inputs
            )

        return render_template('main.html',
                               preds=preds,
                               inputs=inputs
                               )


def predict(model, scaler, inputs):

    data = [[
             inputs['medianIncome'],
             inputs['housingMedianAge'],
             inputs['totalRooms'],
             inputs['totalBedrooms'],
             inputs['population'],
             inputs['households'],
             inputs['latitude'],
             inputs['longitude']
            ]]
    data = scaler.transform(data)
    prediction = model.predict(data)
    prediction = int(prediction.item()*1e04)

    return prediction


if __name__ == '__main__':
    app.run(debug=True)
