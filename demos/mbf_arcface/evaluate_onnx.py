import onnxruntime as ort
from PIL import Image
import numpy as np

def preprocess_image(image_path, input_shape):
    # Open the image file
    image = Image.open(image_path).convert('RGB')
    # Resize the image to the required input shape
    image = image.resize((input_shape[2], input_shape[3]))
    # Convert image to numpy array
    image_data = np.array(image).astype('float32')
    # Normalize the image
    image_data = image_data / 255.0
    # Transpose the image to match the model input shape (1, 3, H, W)
    image_data = np.transpose(image_data, (2, 0, 1))
    # Add batch dimension
    image_data = np.expand_dims(image_data, axis=0)
    return image_data

def evaluate_image_with_onnx(image_path, onnx_model_path):
    # Load the ONNX model
    ort_session = ort.InferenceSession(onnx_model_path)

    # Get the model input details
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape

    # Preprocess the image
    image_data = preprocess_image(image_path, input_shape)

    # Run inference
    outputs = ort_session.run(None, {input_name: image_data})

    # Print the outputs
    # for i, output in enumerate(outputs):
    #     print(f"Output {i}:")
    #     print(output)
    return outputs[0]

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate an image with an ONNX model.')
    parser.add_argument('onnx_model_path', type=str, help='Path to the ONNX model.')
    parser.add_argument('image1_path', type=str, help='Path to the input image.')
    parser.add_argument('image2_path', type=str, help='Path to the input image.')

    args = parser.parse_args()

    # Evaluate the image
    img1_feat = evaluate_image_with_onnx(args.image1_path, args.onnx_model_path)
    img2_feat = evaluate_image_with_onnx(args.image2_path, args.onnx_model_path)

    # Compute the cosine similarity between the two features
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(img1_feat, img2_feat)
    print(f"Cosine similarity: {similarity[0][0]}")
