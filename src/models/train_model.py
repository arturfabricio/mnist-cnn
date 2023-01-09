import argparse
import sys

import torch
import click
from model import MyAwesomeModel
from sklearn.model_selection import train_test_split
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the path to data.py to the system path
data_path = os.path.join(os.path.dirname(__file__), '../data')
sys.path.append(os.path.abspath(data_path))

# Import the data module
from data import mnist

@click.group()
def cli():
    pass

@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=10, help='number of epcohs to use for training' )
def train(lr,epochs):
    print("Training MNIST dataset...")
    print("Learning rate: ",lr)
    print("Number of epochs: ", epochs)

    model = MyAwesomeModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use cuda or cpu
    print("Used device: ", device)
    model.to(device)
    
    train_dl, val_dl = mnist()

    images, labels = next(iter(val_dl))
    ps = torch.exp(model(images))
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print("Epoch number: " + str(epoch))
        train_losses, test_losses = [], []
        running_loss = 0
        
        model.train()
        for inputs, targets in train_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # validate the model
        model.eval()  # switch to evaluation mode
        with torch.no_grad():
            accuracy = 0
            valid_loss = 0
            for images, labels in val_dl:
                log_ps = model(images)
                valid_loss += loss_fn(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

            valid_loss /= len(val_dl)
            accuracy /= len(val_dl)
        model.train()  # switch back to training mode
        
        train_losses.append(running_loss/len(train_dl))
        test_losses.append(valid_loss)

        print(f'Epoch: {epoch+1}/{epochs}.. ',
            f'Train loss: {running_loss/len(train_dl):.3f}.. ',
            f'Valid loss: {valid_loss:.3f}.. ',
            f'Valid accuracy: {accuracy.item()*100}%')

    torch.save(model.state_dict(), 'my_model.pth')

    # plt.plot(train_losses, label="Training loss")
    # plt.xlabel("Training step")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()

@click.command()
@click.argument("model_checkpoint")

def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    