from flask import Flask, render_template, jsonify, flash, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home_page():
    return render_template("index.html"), 200


if __name__ == '__main__':
    app.run(port=3000, debug=True)