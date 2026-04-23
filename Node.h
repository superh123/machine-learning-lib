#pragma once

#include "Matrix.h"
#include <stdexcept>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_set>
#include <stack>

// Using shared ptr, since we'll have multiple connections in our graph
class Node;
typedef std::shared_ptr<Node> Nodeptr;

/**
 * Node in a computational graph for automatic differentiation
 *
 * This is the fundamental building block for neural networks.
 * Each Node represents:
 * - A tensor (Matrix) value
 * - Its gradient (how loss changes with respect to this value)
 * - The operation that created it (e.g., "matmul", "add", "sigmoid")
 * - Connections to previous nodes (the computation DAG)
 *
 * When you perform operations on Nodes, they create new Nodes
 * and record the operation, enabling automatic backpropagation.
 *
 * Example:
 *   Node* a = new Node(X);           // input
 *   Node* b = new Node(W);           // weights
 *   Node* c = matmul(a, b);          // creates new Node with operation="matmul"
 *   c->backward();                   // computes gradients for a and b
 */
class Node : public std::enable_shared_from_this<Node>
// NOTE: Inherit from enable_shared_from_this so we can return a copy of a pre-existing shared pter

{

public:
    static Nodeptr create(const Matrix &data, const std::string &operation = "", bool requires_grad = false);

    /**  Primitive functions */
    static Nodeptr add(const Nodeptr &n1, const Nodeptr &n2);
    static Nodeptr subtract(const Nodeptr &n1, const Nodeptr &n2);
    static Nodeptr multiply(const Nodeptr &n1, const Nodeptr &n2);
    static Nodeptr power(const Nodeptr &n1, double power);
    static Nodeptr divide(const Nodeptr &n1, double rhs);

    /**  Activation functions */
    static Nodeptr relu(const Nodeptr &input);
    static Nodeptr tanh(const Nodeptr &input);
    static Nodeptr sigmoid(const Nodeptr &input);
    static Nodeptr leaky_relu(const Nodeptr &input);
    static Nodeptr mse(const Nodeptr &y_true, const Nodeptr &y_pred);
    static Nodeptr bce(const Nodeptr &y_true, const Nodeptr &y_pred);
    static Nodeptr softmaxCE(const Nodeptr &y_true, const Nodeptr &y_pred);

    /** Calls sum operation of Matrix class for node */
    static Nodeptr sum(const Nodeptr &n1);

    /** Initiate backpropagation */
    void backprop();

    /** Construct topological graph for dependency ordering */
    static void constructTopo(std::shared_ptr<Node> n,
                              std::stack<Nodeptr> &topoSorted,
                              std::unordered_set<Nodeptr> &visited);

    /** Internally used in primitive and activation functions
     *  to propagate gradients to children
     */
    std::function<void()> former;

    inline void scaleGradient(double num) { data.scale(num); }

    std::vector<size_t> getDataDimensions()
    {
        return (this->data.getDimensions());
    }

    /** Calls generatePseudoRand on Matrix data */
    void generatePseudoRand(double start, double end);

    void zeroGradient();

    Matrix &getGrad() { return this->gradient; }

    Matrix &getData() { return this->data; }

    void setData(Matrix other) { this->data = other; }

    void toString() const;

    bool requires_grad; // FIXME: AS OF NOW, NOT USED, REMOVE OR UPDATE?

private:
    Matrix data;               // Forward value
    Matrix gradient;           // Backward gradient (∂L/∂data)
    std::string operation;     // "matmul", "add", "sigmoid", "relu", etc.
    std::vector<Nodeptr> prev; // Parent nodes in computation graph

    Node(const Matrix &data, const std::string &operation = "", bool requires_grad = false);

    // Used internally only for now in softmax cross entropy loss func
    static Matrix softmax(const Nodeptr &logitsNode);
};
