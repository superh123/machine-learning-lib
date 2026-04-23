#pragma once

#include "Matrix.h"
#include "Node.h"
#include "Activation.h"
#include <random>

/**
 * Layer - Fully connected neural network layer
 *
 * Performs: output = activation(input × weights + bias)
 *
 * Built on the autograd Node system:
 * - weights and bias are trainable Node objects
 * - forward() builds computation graph
 * - backward() flows automatically via Node graph
 *
 * Example:
 *   Layer layer(2, 4, "sigmoid");  // 2 inputs → 4 outputs
 *   Nodeptr out = layer.forward(input);
 *   loss->backward();  // gradients flow to layer's weights/bias
 *
 * NOTES:
 * Input (Batch size x Input features)
 * Weights (Input features x Output neurons)
 * Output (Batchsize x output neurons)
 * Bias broadcast to each neuron (1 x output neurons)
 */
class Layer
{
private:
    // Weights matrix wrapped in Node object for backprop
    Nodeptr weights;

    // To be clear, this isn't a single 'bias,' it's a matrix that holds multiple biases
    // depending on the number of output neurons in the calculation. Perhaps bias matrix
    // would be more adequate to describe it
    Nodeptr bias;

public:
    Layer(size_t inputFeatures, size_t numNeurons, const ActivationType &act_t);

    /** Build computation graph*/
    Nodeptr forward(Nodeptr input);

    /** Zero out weights and bias gradients
        for next epoch, to prevent gradient accumulation
    */
    void zeroGradients();

    /** Update weight and bias parameters (data) */
    void update(double learningRate);

    void printParam();

    ActivationType act_t;
};