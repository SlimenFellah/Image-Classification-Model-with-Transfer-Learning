from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from model import SimpleCNN

app = Flask(__name__)

# Initialize the model
model = SimpleCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(io.BytesIO(file.read()))
        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            label = class_labels[predicted.item()]

        return jsonify({'prediction': label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
