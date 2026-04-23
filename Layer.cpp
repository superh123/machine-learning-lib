#include "Matrix.h"
#include "Node.h"
#include "Layer.h"
#include "Activation.h"
#include <random>
#include <assert.h>

Layer::Layer(size_t inputFeatures, size_t numNeurons, const ActivationType &act_t) : act_t(act_t)
{
    weights = Node::create(Matrix(inputFeatures, numNeurons), "", true);
    weights->generatePseudoRand(-1, 1); // Generate random values for all weight parameters in weights matrix
    bias = Node::create(Matrix(1, numNeurons), "", true);
}

Nodeptr Layer::forward(Nodeptr input)
{

    std::vector<size_t> weightsDim = weights->getDataDimensions();
    std::vector<size_t> inputsDim = input->getDataDimensions();

    size_t columnsX = inputsDim.at(1); // Columns of input
    size_t rowsW = weightsDim.at(0);   // Rows of weights matrix

    if (columnsX != rowsW)
    {
        throw std::invalid_argument("Input tensor is not valid size for calculation");
    }

    // Layer will follow Z = XW + b updates

    Nodeptr sum = Node::multiply(input, weights); // Matrix multiply XW

    sum = Node::add(sum, bias); // Add bias elmentwise to form output neuron

    return sum;
}

void Layer::update(double learningRate)
{

    // Update formula : W = Wdata - n * Wgrad
    // NOTE: Maybe add optimizers/momentum?

    Matrix updatedWeights = weights->getData() - (weights->getGrad() * learningRate);

    weights->setData(updatedWeights);

    Matrix updatedBias = bias->getData() - (bias->getGrad() * learningRate);

    bias->setData(updatedBias);
}

void Layer::zeroGradients()
{
    weights->zeroGradient();
    bias->zeroGradient();
}

void Layer::printParam()
{
    std::cout << weights->getData().getRows() << "x" << weights->getData().getCols() << " ";
}