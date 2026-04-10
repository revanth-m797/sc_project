from flask import Flask, request, jsonify, render_template
from test1 import main
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')  # serves your HTML

@app.route('/find-route', methods=['POST'])
def find_route():
    data = request.json
    source = data.get('source')
    dest = data.get('destination')
    pref =data.get("pref")

    result = main(source, dest, pref)


    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)