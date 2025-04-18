# DA6401 - Assignment 1

MD Karimulla Haque MA23C021

Instructions to train and evaluate the neural network models:

To train a neural network model for image classification on the Fashion-MNIST dataset using categorical cross-entropy loss, import network.py, and use NeuralNetwork with any configuration to create the neural network.
To train a neural network model:

use NeuralNetwork.{optimizer}
optimizer: optimization routine 
 (Normal, Momentum, Nesterov, AdaGrad, RMSProp, Adam, Nadam)

Link to the project report: https://api.wandb.ai/links/mdkarimullahaque-iit-madras/amny3btr

The NN training framework:

These codes are based on a procedural framework and make no use of classes for NN models like keras does for the simplicity of understanding the code. This code works only for classification tasks and by default assumes that the activation function for the last layer is softmax.

Training NN:

To train the NN, it takes the training data, the validation data and the hyperparameters and Trains a NN specified by size_hidden_layer and num_hidden_layers. This code provides flexibility in choosing the following hyperparameters:
        
        'num_epochs': 10, 50
        'num_hidden_layers': 3, 4, 5
        'size_hidden_layer': 32, 64, 128
        'weight_decay': 0, 0.0005, 0.5
        'learning_rate': 1e-3, 1e-4
        'optimizer':"Normal","Momentum","AdaGrad","RMSProp","Adam","Nadam"
        'batch_size': 128, 1024, 60000
        'weight_init':  "RandomNormal", "XavierUniform"
        'activation': "Sigmoid", "Tanh", "Relu"
       


