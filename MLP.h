#pragma once

#include "Matrix.h"
#include "Node.h"
#include "Layer.h"
#include "Activation.h"
#include <random>

/**
 * MLP - Multi-Layer Perceptron
 *
 * Sequential container of Layer objects that form a complete neural network.
 *
 * Forward pass: data flows through layers in order
 * Backward pass: gradients flow backwards through layers via autograd
 *
 * Example:
 *   MLP mlp(2, {4, 1});  // 2 inputs → 4 hidden → 1 output
 *   Nodeptr pred = mlp.forward(input);
 *   Nodeptr loss = mse(pred, target);
 *   loss->backward();  // computes gradients for all layers
 *   mlp.update(lr);    // updates all weights/biases
 */

class MLP
{
private:
    std::vector<size_t> sizes; // Used to construct the size of our weight and bias matrices for our layers
    std::vector<Layer> layers; // Number of layers (Hidden + Output) in our MLP

    double learningRate;

public:
    MLP(size_t sizein, std::vector<size_t> outLayerSizes, const double learningRate);
    Nodeptr forward(Nodeptr input);   // Initiate forward pass, learning
    void backward(Nodeptr &loss);     // Reflect on decisions made, adjust weight parameters accordingly to gradients
    void zeroGrad();                  // Zero out gradients for next epoch
    void update(double learningRate); // Update weight and bias matrix parameterss
    void printParam();
};