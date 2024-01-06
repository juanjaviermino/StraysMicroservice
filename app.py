import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from google.cloud import storage
from google.oauth2 import service_account
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Middleware for interacting with your React serve

# Carga del modelo pre entrenado RestNet50
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Authenticate using your service account key file
credentials = service_account.Credentials.from_service_account_file('silicon-brace-410116-7c097b027311.json')
storage_client = storage.Client(credentials=credentials)
bucket_name = 'straysimagesbucket'

def extract_features(img_url, model):
    # Extract the file name from the URL
    file_name = img_url.split('/')[-1]

    # Access the bucket and the file
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    img_bytes = blob.download_as_bytes()

    # Load the image from bytes
    img = Image.open(BytesIO(img_bytes))
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
        try:
            image_features = extract_features(file_path, model)
            similarity = cosine_similarity(base_image_features, image_features)[0][0]
            similarities[file_path] = similarity
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

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
        top_matches = [{'imageURL': img, 'similitud': float(sim)} for img, sim in top_matches]

        return jsonify({"Matches": top_matches}), 200
    except Exception as e:
        return jsonify({"Error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)

