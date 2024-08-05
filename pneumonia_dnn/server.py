"""
Flask API Entrypoint
"""

from flask import Flask

from pneumonia_dnn.controllers.dataset_controller import dataset_blueprint
from pneumonia_dnn.controllers.project_controller import project_blueprint


app = Flask(__name__)

app.register_blueprint(dataset_blueprint, url_prefix="/dataset")
app.register_blueprint(project_blueprint, url_prefix="/project")

if __name__ == "__main__":
    app.run(debug=True)
