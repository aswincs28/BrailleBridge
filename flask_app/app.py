import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request
from model.infer import predict_braille


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    if request.method == "POST":
        img = request.files["image"]
        save_path = os.path.join("static", img.filename)
        img.save(save_path)
        text = predict_braille(save_path)

    return render_template("index.html", text=text)


if __name__ == "__main__":
    app.run(debug=True)
