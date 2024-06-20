# Image Classification with Transfer Learning

## Project Description

This project demonstrates an advanced image classification model using transfer learning with a pre-trained ResNet50 model. The model is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes. This project covers various stages, including data preprocessing, model training, evaluation, and deployment using a Flask web application.

## Features

1. **Data Preprocessing and Augmentation**: Includes advanced data augmentation techniques to enhance the training dataset.
2. **Transfer Learning**: Utilizes a pre-trained ResNet50 model for feature extraction and adds custom layers for classification.
3. **Model Evaluation**: Detailed evaluation using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
4. **Model Explainability**: Visualizes which parts of an image contribute most to the model's predictions using Grad-CAM.
5. **Deployment**: Deploys the trained model using a Flask web application for real-time image classification.

## Installation

To get started, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/SlimenFellah/Image-Classification-Model-with-Transfer-Learning.git
cd image-classification-with-transfer-learning
pip install -r requirements.txt
```

## Running the Project

### Step 1: Exploratory Data Analysis (EDA)

Perform EDA to understand the dataset better and visualize the images and class distribution.

```bash
python eda.py
```

### Step 2: Train the Model

Train the image classification model using the CIFAR-10 dataset. This step includes data augmentation, transfer learning, and hyperparameter tuning.

```bash
python model.py
```

### Step 3: Evaluate the Model

Evaluate the trained model using various metrics and visualize the results.

```bash
python evaluation.py
```

### Step 4: Deploy the Model

Deploy the model using a Flask web application. This allows you to upload images and get real-time classification results.

```bash
python deployment.py
```

The Flask app will be available at `http://127.0.0.1:5000`.

## Usage

### Using the Deployed Model

1. Start the Flask application:

    ```bash
    python deployment.py
    ```

2. Use a tool like `curl` or Postman to send a POST request to the `/predict` endpoint with an image file.

    ```bash
    curl -X POST -F "file=@path/to/your/image.jpg" http://127.0.0.1:5000/predict
    ```

    You should receive a JSON response with the predicted class of the image.

## Project Structure

- `main.py`: Main script to run the entire project.
- `eda.py`: Script for exploratory data analysis.
- `model.py`: Script to define and train the model.
- `evaluation.py`: Script for model evaluation and visualization.
- `deployment.py`: Script to deploy the model using Flask.
- `requirements.txt`: File containing the list of dependencies.

## Requirements

- torch
- torchvision
- Flask
- numpy
- matplotlib
- seaborn
- scikit-learn
- Pillow

## Acknowledgements

This project utilizes the CIFAR-10 dataset, obtained from the torchvision library, and the ResNet50 model pre-trained on ImageNet, also from torchvision.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.