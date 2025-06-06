import tempfile
import os
import uuid
from flask import Flask, jsonify, render_template, send_file, redirect, request
from werkzeug.utils import secure_filename
from googletrans import Translator
import cv2
import pytesseract
from PIL import Image

from OBR import SegmentationEngine, BrailleClassifier, BrailleImage

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
tempdir = tempfile.TemporaryDirectory()

app = Flask("Optical Braille Recognition Demo")
app.config['UPLOAD_FOLDER'] = tempdir.name

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/favicon.ico')
def fav():
    return send_file('favicon.ico', mimetype='image/ico')

@app.route('/coverimage')
def cover_image():
    return send_file('samples/sample1.png', mimetype='image/png')

@app.route('/procimage/<string:img_id>')
def proc_image(img_id):
    global tempdir
    print(img_id)
    image = '{}/{}-proc.png'.format(tempdir.name, secure_filename(img_id))
    if os.path.exists(image) and os.path.isfile(image):
        return send_file(image, mimetype='image/png')
    return redirect('/coverimage')

@app.route('/digest', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": True, "message": "file not in request"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": True, "message": "empty filename"})
    if file and allowed_file(file.filename):
        filename = ''.join(str(uuid.uuid4()).split('-'))
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        global tempdir
        
        image_path = '{}/{}'.format(tempdir.name, filename)
        classifier = BrailleClassifier()
        img = BrailleImage(image_path)
        for letter in SegmentationEngine(image=img):
            letter.mark()
            classifier.push(letter)
        cv2.imwrite('{}/{}-proc.png'.format(tempdir.name, filename), img.get_final_image())
        os.unlink(image_path)

        r = {
            "error": False,
            "message": "Processed and Digested successfully",
            "img_id": filename,
            "digest": classifier.digest()
        }
        return jsonify(r)

@app.route('/translate', methods=['POST'])
def translate_text():
    # Get the English text from the request
    english_text = request.json.get('text', '')
    target_lang = request.json.get('language', 'te')  # Default is Telugu ('te')

    # Validate the provided text
    if not english_text:
        return jsonify({"error": True, "message": "No text provided for translation"})

    # Validate the provided language code
    if target_lang not in ['te', 'hi']:  # 'te' for Telugu, 'hi' for Hindi
        return jsonify({"error": True, "message": "Invalid language code. Use 'te' for Telugu or 'hi' for Hindi."})

    # Initialize the Google Translate API
    translator = Translator()

    try:
        # Translate the text to the specified language (Telugu or Hindi)
        translated_text = translator.translate(english_text, src='en', dest=target_lang).text

        # Return the translated text
        return jsonify({"error": False, "translated_text": translated_text})

    except Exception as e:
        return jsonify({"error": True, "message": f"Translation failed: {str(e)}"})


@app.route('/convert', methods=['POST'])
def convert_image_to_text():
    if 'file' not in request.files:
        return jsonify({'error': True, 'message': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        # Save and process the image
        img = Image.open(file)
        # Use pytesseract for OCR (replace with Braille-to-English logic if needed)
        text = pytesseract.image_to_string(img)
        return jsonify({'error': False, 'digest': text, 'message': 'Conversion successful'})
    except Exception as e:
        return jsonify({'error': True, 'message': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
    tempdir.cleanup()