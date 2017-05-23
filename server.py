import time

from flask import Flask, request
from flask_cors import CORS

from classifier import Classifier


inst = Classifier()
app = Flask(__name__)
cors = CORS(app)
@app.route("/recognize", methods=['POST'])
def predict():
    start = time.time()

    data = request.data.decode("utf-8")

    print request.files['image']

    value = inst.getNumber(request.files['image'])
    print("Time spent handling the request: %f" % (time.time() - start))
    return value

if __name__ == "__main__":
    print('Starting the API')
    app.run()