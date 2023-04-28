from flask import Flask, request, jsonify

app = Flask(__name__)

# Routes
# Home page routegr
@app.route("/")
def home():
    return "Hello, This is FlaskAPI!"
