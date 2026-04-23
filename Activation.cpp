#include "Node.h"
#include "Matrix.h"
#include "Activation.h"

/**
 * Defines different activation types to use
 * on neurons
 */

Nodeptr Activation::Relu(const Nodeptr &val)
{
    return Node::relu(val);
}

Nodeptr Activation::Sigmoid(const Nodeptr &val)
{
    return Node::sigmoid(val);
}

Nodeptr Activation::Tanh(const Nodeptr &val)
{
    return Node::tanh(val);
}

Nodeptr Activation::LeakyRelu(const Nodeptr &val)
{
    return Node::leaky_relu(val);
}
