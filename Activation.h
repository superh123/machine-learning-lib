#pragma once

#include "Node.h"

enum ActivationType
{
    RELU,
    SIGMOID,
    TANH,
    LEAKY_RELU
};

class Activation
{
private:
    static Nodeptr Relu(const Nodeptr &val);
    static Nodeptr Sigmoid(const Nodeptr &val);
    static Nodeptr Tanh(const Nodeptr &val);
    static Nodeptr LeakyRelu(const Nodeptr &val);

public:
    // Allows us to grab the activation that's needed quickly
    static inline std::unordered_map<ActivationType, std::function<Nodeptr(const Nodeptr &)>> nActivationFnc =
        {
            {ActivationType::RELU, Relu},
            {ActivationType::SIGMOID, Sigmoid},
            {ActivationType::TANH, Tanh},
            {ActivationType::LEAKY_RELU, LeakyRelu}};
};
