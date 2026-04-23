#include "Matrix.h"
#include "Node.h"
#include "Layer.h"
#include "MLP.h"
#include "Activation.h"
#include <random>
#include <vector>

/**
 * Construct a multi-layer perceptron
 * @param sizein - Represents the input features of the first layers weights (must equal inputs input features)
 * @param outLayers - How many layers to be constructed
 * @param learningRate - How aggressively do you want the model to learn?
 *
 * NOTE: sizein and outLayers are combined to form the neural networks
 *       layer dimensions (sizes param in layer)
 *
 * EXAMPLE: Say input is 4x2
 *          MLP(2, {1}) forms a single output layer 2 x 1.
 *          MLP(2, {4, 1}) forms a hidden layer 2x4 and output layer 4x1
 *          You must ensure your output layers dimensions match your intended outputs
 */
MLP::MLP(size_t sizein, std::vector<size_t> outLayers, const double learningRate) : learningRate(learningRate)
{
    sizes.reserve(1 + outLayers.size());
    sizes.emplace_back(sizein);
    sizes.insert(sizes.end(), outLayers.begin(), outLayers.end());
    for (size_t i = 0; i < sizes.size() - 1; i++)
    {
        layers.emplace_back(sizes.at(i), sizes.at(i + 1), ActivationType::RELU);
    }
}

void MLP::zeroGrad()
{
    for (auto &layer : layers)
    {
        layer.zeroGradients();
    }
}

Nodeptr MLP::forward(Nodeptr input)
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        input = layers[i].forward(input);

        /**
         * NOTE: For MNIST training, leave last layer activation empty, else use RELU
         *       For XOR training, set SIGMOID activation for both if and else
         *
         */

        if (i == layers.size() - 1) // last layer activation
        {
            // input = Activation::nActivationFnc[ActivationType::SIGMOID](input);
        }
        else
        {
            input = Activation::nActivationFnc[ActivationType::RELU](input);
        }
    }

    return input;
}

void MLP::backward(Nodeptr &loss)
{
    loss->backprop();

    for (Layer &layer : layers)
    {
        layer.update(learningRate);
    }

    this->zeroGrad();
}

void MLP::update(double learningRate)
{

    for (Layer &layer : layers)
    {
        layer.update(learningRate);
    }
}

void MLP::printParam()
{

    for (Layer layer : layers)
    {
        layer.printParam();
    }
}