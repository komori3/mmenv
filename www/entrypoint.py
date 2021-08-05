#!/usr/bin/python3

import flask
import tasks

app = flask.Flask(__name__)
app.register_blueprint(tasks.app)

@app.route('/')
def home():
    return flask.render_template('home.html', title='Home')

if __name__ == "__main__":
    app.run(debug=True)