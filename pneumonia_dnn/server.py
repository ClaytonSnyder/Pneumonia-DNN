"""
Flask API Entrypoint
"""

from flask import Flask

from pneumonia_dnn.controllers.dataset_controller import dataset_blueprint


app = Flask(__name__)

app.register_blueprint(dataset_blueprint, url_prefix="/dataset")

if __name__ == "__main__":
    app.run(debug=True)
