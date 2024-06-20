# Import necessary modules
from eda import perform_eda
from model import train_model
from evaluation import evaluate_model
from deployment import run_app

def main():
    # Perform EDA
    perform_eda()

    # Train the model
    model = train_model()

    # Evaluate the model
    evaluate_model(model)

    # Run the deployment
    run_app(model)

if __name__ == "__main__":
    main()
