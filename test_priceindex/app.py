import datetime
import pandas as pd
from flask import Flask, request, jsonify
from solgt.priceindex.priceindex import Priceindex

app = Flask(__name__)


# Home page route
@app.route("/")
def home():
    return "Hello! Flask!"


# Priceindex route
@app.route("/pi/", methods=['POST'])
def get_priceindex():
    # return "Hello! Priceindex!"

    pi = Priceindex()
    dates = pi.date.tolist()


    res = {"date": (pi.date).tolist(), "price": (pi.price).tolist()}

    # Return the response
    return jsonify(res)
