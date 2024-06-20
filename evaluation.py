import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import SimpleCNN

def evaluate_model(model, testloader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    average_loss = test_loss / len(testloader)
    
    print(f'Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    return average_loss, accuracy

def main():
    # Load the CIFAR-10 test dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Initialize the model
    model = SimpleCNN()
    
    # Load the trained model weights
    model.load_state_dict(torch.load('model.pth'))
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate the model
    evaluate_model(model, testloader, criterion)

if __name__ == "__main__":
    main()
