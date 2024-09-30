from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import io

app = Flask(__name__)

# Initialize the object detection pipeline with a specific model
object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Ensure an image file was provided in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Read the image from the request
    image_file = request.files['image']
    try:
        image = Image.open(io.BytesIO(image_file.read()))
    except Exception as e:
        return jsonify({'error': f'Invalid image format: {str(e)}'}), 400

    # Perform object detection
    bounding_boxes = object_detector(image)

    # Prepare the results
    results = []
    for index, bounding_box in enumerate(bounding_boxes):
        label = bounding_box["label"]
        box = bounding_box["box"]

        # Extract bounding box dimensions
        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]

        # Calculate object size
        width = xmax - xmin
        height = ymax - ymin

        # Append each detected object to the results
        results.append({
            "object": index + 1,
            "label": label,
            "bounding_box": {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "width": width,
                "height": height
            }
        })

    return jsonify({"detected_objects": results})

if __name__ == '__main__':
    app.run(debug=True)
