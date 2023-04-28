from flask import Flask

app = Flask(__name__)

# Routes
# Home page route
@app.route("/")
def home():
    return "Hello, Flask!"
