from flask import Flask, render_template, request
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

modelo = MobileNetV2(weights="imagenet")

@app.route("/")
def inicio():
    return render_template("index.html")

@app.route("/clasificar", methods=["POST"])
def clasificar():

    archivo = request.files["imagen"]

    ruta = os.path.join(
        app.config["UPLOAD_FOLDER"],
        archivo.filename
    )

    archivo.save(ruta)

    img = image.load_img(
        ruta,
        target_size=(224, 224)
    )

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    pred = modelo.predict(x)

    resultado = decode_predictions(
        pred,
        top=1
    )[0][0]

    nombre = resultado[1]
    confianza = round(resultado[2] * 100)

    if "cat" in nombre.lower():
        clase = "🐱 Gato"

    elif (
        "dog" in nombre.lower()
        or "pug" in nombre.lower()
        or "beagle" in nombre.lower()
        or "labrador" in nombre.lower()
        or "retriever" in nombre.lower()
        or "terrier" in nombre.lower()
    ):
        clase = "🐶 Perro"

    else:
        clase = f"🔍 {nombre}"

    return render_template(
        "index.html",
        imagen=ruta,
        resultado=clase,
        confianza=confianza
    )

if __name__ == "__main__":
    app.run(debug=True, port=5001)