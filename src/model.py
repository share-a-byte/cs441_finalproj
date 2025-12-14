import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class AudioCNN(nn.Module):
    def __init__(self, num_classes=2, dropout = 0.5):
        super(AudioCNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(5,5), padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
			      nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
		)
    def forward(self, x):
        return self.classifier(x)
    
def evaluate_model(model, loader):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print('Device is {}'.format(device))
    N = 0; accuracy = 0; loss = 0
    loss_function = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(False): 
        for _, data in enumerate(loader):
            inputs, targets = data
            N += len(targets)
            outputs = model(inputs.to(torch.float32).to(device))
            accuracy += sum(outputs.cpu().numpy() == targets.numpy())
            loss += loss_function(outputs, targets.to(device)).item() * len(targets)
        return loss/N, 1-accuracy/N
    
def display_error_curves(training_losses, validation_losses):
    num_epochs = len(training_losses)
    plt.plot(range(num_epochs), training_losses, label="Training Loss")
    plt.plot(range(num_epochs), validation_losses, label="Validation Loss")
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

def train_model(model, train_loader, val_loader, num_epochs, lr):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print('Device is {}'.format(device))
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    val_loss = []
    torch.manual_seed(1)
    for epoch in range(num_epochs):
        curr_loss = 0.0
        N = 0
        for _, data in enumerate(train_loader):
            inputs, targets = data
            #shape is [ninputs, nchannels, spec height, spec width]
            optimizer.zero_grad()
            print('Input tensor shape: {}'.format(inputs.shape))
            outputs = model(inputs.to(torch.float32).to(device))
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            curr_loss+=loss.item()*len(targets)
            N+= len(targets)
        curr_loss = curr_loss/N
        train_loss.append(curr_loss)
        #TODO: Calculate val loss and print
        e_val_loss, e_val_err = evaluate_model(model, val_loader)
        val_loss.append(e_val_loss)
        print('>>>EPOCH {}<<<\n   Train loss: {}   Val Loss: {}   Error: {}\n'.format(epoch, curr_loss, e_val_loss, e_val_err))
    f_val_loss, f_val_err = evaluate_model(model, val_loader)
    print('>>>FINAL Val loss: {} Error: {}\n'.format(f_val_loss, f_val_err))
    display_error_curves(train_loss, val_loss)
      
def inference(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print('Device is {}'.format(device))
    correct = 0; total = 0
    with torch.no_grad():
        for _, data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            # Count of predictions that matched the target label
            correct += (outputs == labels).sum().item()
            total += outputs.shape[0]
        acc = correct/total
        print(f'Accuracy: {acc:.2f}, Total items: {total}')



   
   