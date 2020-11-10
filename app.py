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
        inputs['MedInc'] = request.form['MedInc']
        inputs['HouseAge'] = request.form['HouseAge']
        inputs['AveRooms'] = request.form['AveRooms']
        inputs['AveBedrms'] = request.form['AveBedrms']
        inputs['Population'] = request.form['Population']
        inputs['AveOccup'] = request.form['AveOccup']
        inputs['Latitude'] = request.form['Latitude']
        inputs['Longitude'] = request.form['Longitude']

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
             inputs['MedInc'],
             inputs['HouseAge'],
             inputs['AveRooms'],
             inputs['AveBedrms'],
             inputs['Population'],
             inputs['AveOccup'],
             inputs['Latitude'],
             inputs['Longitude']
            ]]
    data = scaler.transform(data)
    prediction = model.predict(data)

    return prediction.item()


if __name__ == '__main__':
    app.run(debug=True)
