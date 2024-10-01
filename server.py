import sys
sys.path.insert(0, './lib')

from flask import Flask, jsonify, request
from hibrid import predictPrice


app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    token = request.args.get('token')
    days = int(request.args.get('days'))
    price, date, _ = predictPrice()  # replace with your actual function
    return jsonify({'token': token, 'predicted_price': price, 'date': date.strftime('%Y-%m-%d')})


if __name__ == '__main__':
    app.run(debug=True, port=3000)
