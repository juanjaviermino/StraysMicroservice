import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Carga del modelo pre entrenado RestNet50
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path, model):
    """Extracción de características de la imagen"""
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features

def compare_images(base_image_path, image_paths):
    """Compara la imagen base con un conjunto de imágenes"""
    base_image_features = extract_features(base_image_path, model)
    similarities = {}

    for file_path in image_paths:
        if os.path.isfile(file_path) and file_path != base_image_path:
            image_features = extract_features(file_path, model)
            similarity = cosine_similarity(base_image_features, image_features)[0][0]
            similarities[file_path] = similarity

    # Sort by similarity
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_similarities[:3]

@app.route('/compare', methods=['POST'])
def compare_route():
    data = request.json
    base_image = data.get('base_image')
    compare_images_list = data.get('compare_images')

    # Validate inputs
    if not base_image or not isinstance(compare_images_list, list):
        return jsonify({"error": "Invalid input data"}), 400

    # Perform image comparison
    try:
        top_matches = compare_images(base_image, compare_images_list)
        print(top_matches)

        # Convert numpy.float32 to Python float for JSON serialization
        top_matches = [(img, float(sim)) for img, sim in top_matches]

        return jsonify({"Matches": top_matches}), 200
    except Exception as e:
        return jsonify({"Error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)

