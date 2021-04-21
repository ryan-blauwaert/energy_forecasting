# Demonstrates Bootstrap version 3.3 Starter Template
# available here: https://getbootstrap.com/docs/3.3/getting-started/#examples

from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)

# home page
@app.route('/')
def index():
    region = ['US48', 'CAL', 'CAR', 'CENT', 'FLA', 'MIDA', 'MIDW', 'NE',
                 'NY', 'NW', 'SE', 'SW', 'TEN', 'TEX']
    return render_template('index.html', region=region)

@app.route('/forecast')
def forecast():
    region = ['US48', 'CAL', 'CAR', 'CENT', 'FLA', 'MIDA', 'MIDW', 'NE',
                 'NY', 'NW', 'SE', 'SW', 'TEN', 'TEX']
    reg = request.args.get("region")
    return render_template('forecast.html', region=region, reg=reg)

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/about_me')
def about_me():
    return render_template('about_me.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
