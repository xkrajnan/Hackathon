# pylint: disable=trailing-newlines
"""
Starting point for the brand story generation server.

:Authors:
    Vojtech Krajnansky <vojtech.krajnansky@semeai.cz>
"""
# Logging setup
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

# Third-party imports
from flask import Flask
from flask_cors import CORS
from flask_restful import Api

import controller

def create_app():
    """
    Creates the QuerySim Flask app.
    """
    app = Flask(__name__)
    api = Api(app)
    cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Add endpoint controllers
    api.add_resource(controller.BrandStory, '/api/generateBS')

    return app


# Take off!
if __name__ == '__main__':
    APP = create_app()
    APP.run(debug=True, port=5000)

