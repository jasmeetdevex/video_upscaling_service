from flask_pymongo import PyMongo
import os

mongo = PyMongo()

def init_mongo(app):
    app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb+srv://narratixdev:sltgyS9EjJPqfXQw@narratixcluster.00fijbr.mongodb.net/narratixdb?retryWrites=true&w=majority&appName=NarratixCluster")
    mongo.init_app(app)
