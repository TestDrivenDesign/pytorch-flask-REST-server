import os

from flask import Flask, request, jsonify
from numpy import random

app = Flask(__name__)
# from inference import get_skin_prediction

@app.route('/', methods=['GET'])
def basic_get():
    return jsonify({'success': 'success'})
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'no file in request'}), 400
    
    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    

#     # try:
#     #     file=request.files['file']
#     #     image= file.read()
#     #     diagnosis=get_skin_prediction(image_bytes=image)
#     #     print('hits')
#     #     return jsonify({'type': diagnosis}), 200
#     # except:
#     #     return jsonify({'error': 'internal server error'}), 500

    # file=request.files['file']
    # image= file.read()
    # diagnosis=get_skin_prediction(image)
    dummy_result =random.randint(0, 6)
    print(dummy_result)
    
    return jsonify({'type': dummy_result}), 200

    

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('PORT', 3030))

