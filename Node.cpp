#include "Matrix.h"
#include "Node.h"
#include <stdexcept>
#include <memory>
#include <unordered_set>
#include <stack>
#include <cmath>
#include <algorithm>

// FIXME: UPDATE excessive lock statements, with singular at start of former lambda function along with null check

Node::Node(const Matrix &mat, const std::string &op, bool requires_grad)
    : data(mat), operation(op)
{
    this->gradient = this->data;
    this->gradient.fill(0);
}

Nodeptr Node::create(const Matrix &data, const std::string &operation, bool requires_grad)
{
    return std::shared_ptr<Node>(new Node(data, operation, requires_grad));
}

/** ADDITION: C = A + B
 *
 * Forward: Just add element-wise
 * Backward: Gradient flows equally to both inputs
 *   ∂C/∂A = 1, ∂C/∂B = 1
 *   So: grad_A = grad_C, grad_B = grad_C
 *
 *  Reasoning: Since addition is elementwise Cij = Aij + Bij
 *
 *  => ∂Cij/∂Akl = 1 IF (k,l) = (i, j) & ∂Cij/∂Akl = 0 IF (k, l) != (i, j)
 *
 *  => Now testing ∂C11/∂A = [1 0 0 0], ∂C12/∂A = [0 1 0 0]
 *
 *  => Construct Jacobian = [ 1 0 0 0, 0 1 0 0, 0 0 1 0, 0 0 0 1] = dC/dA = IDENTITY MATRIX
 *  => Symbolically represented as 1 in our backward flow as has no impact
 */

Nodeptr Node::add(const Nodeptr &n1, const Nodeptr &n2)
{
    Nodeptr out = create(n1->data + n2->data, "+");
    out->prev = {n1, n2};
    out->former = [n1_weak = std::weak_ptr<Node>(n1),
                   n2_weak = std::weak_ptr<Node>(n2),
                   out_weak = std::weak_ptr<Node>(out)]()
    {
        auto n1ptr = n1_weak.lock();
        auto n2ptr = n2_weak.lock();
        auto outptr = out_weak.lock();

        std::vector<size_t> n1DataDim = n1ptr->data.getDimensions();
        std::vector<size_t> n2DataDim = n2ptr->data.getDimensions();

        if (n1DataDim[0] == n2DataDim[0] && n1DataDim[1] == n2DataDim[1])
        {
            // Normal matrix addition
            n1_weak.lock()->gradient = n1_weak.lock()->gradient + out_weak.lock()->gradient;
            n2_weak.lock()->gradient = n2_weak.lock()->gradient + out_weak.lock()->gradient;
        }
        else if (n2DataDim[0] == 1 && n1DataDim[1] == n2DataDim[1])
        {
            // bias on right (broadcast)
            n1ptr->gradient = n1ptr->gradient + outptr->gradient;
            n2ptr->gradient = n2ptr->gradient + Matrix::columnWiseSum(outptr->gradient);
        }
        else if (n1DataDim[0] == 1 && n1DataDim[1] == n2DataDim[1])
        {
            // bias on left (broadcast)
            n1ptr->gradient = n1ptr->gradient + Matrix::columnWiseSum(outptr->gradient);
            n2ptr->gradient = n2ptr->gradient + outptr->gradient;
        }
        else
        {
            throw std::runtime_error(" Something unexpected happened in add backward ");
        }
    };

    return out;
}

/** * SUBTRACTION: C = A - B
 * * Forward: Just subtract element-wise
 * Backward: Gradient flows forward for A, flipped for B
 * ∂C/∂A = 1,  ∂C/∂B = -1
 * So: grad_A = grad_C, grad_B = -grad_C
 *
 * Reasoning: Since Cij = Aij - Bij
 *
 * => ∂Cij/∂Akl = 1  IF (k,l) = (i, j)
 * => ∂Cij/∂Bkl = -1 IF (k,l) = (i, j)
 *
 * Constructing Jacobians:
 * dC/dA = IDENTITY MATRIX (I)
 * dC/dB = NEGATIVE IDENTITY MATRIX (-I)
 */
Nodeptr Node::subtract(const Nodeptr &n1, const Nodeptr &n2)
{
    Nodeptr out = create(n1->data - n2->data, "-");
    out->prev = {n1, n2};
    out->requires_grad = n1->requires_grad || n2->requires_grad;
    out->former = [n1_weak = std::weak_ptr<Node>(n1),
                   n2_weak = std::weak_ptr<Node>(n2),
                   out_weak = std::weak_ptr<Node>(out)]()
    {
        n1_weak.lock()->gradient = n1_weak.lock()->gradient + (out_weak.lock()->gradient);
        n2_weak.lock()->gradient = n2_weak.lock()->gradient - (out_weak.lock()->gradient);
    };

    return out;
}

/**
 * MATRIX MULTIPLY: C = A × B
 *
 * Forward: Matrix multiplication (A: m×n, B: n×p → C: m×p)
 * Backward: Uses transpose rules
 *   ∂C/∂A = grad_C × B^T
 *   ∂C/∂B = A^T × grad_C
 *
 * Example: If A is 2×3, B is 3×4, C is 2×4
 *   grad_A will be 2×3, grad_B will be 3×4
 */

Nodeptr Node::multiply(const Nodeptr &n1, const Nodeptr &n2)
{
    Nodeptr out = create(n1->data * n2->data, "*");
    out->prev = {n1, n2};
    out->former = [n1_weak = std::weak_ptr<Node>(n1),
                   n2_weak = std::weak_ptr<Node>(n2),
                   out_weak = std::weak_ptr<Node>(out)]()
    {
        n1_weak.lock()->gradient = n1_weak.lock()->gradient + (out_weak.lock()->gradient * Matrix::transpose((n2_weak.lock()->data)));

        n2_weak.lock()->gradient = n2_weak.lock()->gradient + (Matrix::transpose((n1_weak.lock()->data)) * out_weak.lock()->gradient);
    };
    return out;
}

/**
 * POWER OPERATION: C = A ^ p (element-wise exponentiation)
 *
 * Forward: Raises each element of matrix A to the power p
 *   C[i,j] = A[i,j] ^ p
 *
 * Backward: Power rule from calculus
 *   ∂C/∂A = p * A^(p-1)
 *   So: grad_A = grad_C * (p * A^(p-1))
 *
 * Special cases:
 *   p = 0: C = 1 (constant) → grad_A = 0
 *   p = 1: C = A → grad_A = grad_C
 *   p = 2: C = A² → grad_A = grad_C * (2*A)
 */

Nodeptr Node::power(const Nodeptr &n1, double power)
{
    Nodeptr out = create(n1->data ^ power, "^");
    out->prev = {n1};
    out->former = [n1_weak = std::weak_ptr<Node>(n1),
                   out_weak = std::weak_ptr<Node>(out),
                   power]()
    {
        n1_weak.lock()->gradient = n1_weak.lock()->gradient +
                                   Matrix::hadamardProd(out_weak.lock()->gradient,
                                                        ((n1_weak.lock()->data ^ (power - 1)) * (power)));
    };
    return out;
}

/**
 * Division is just scalar multiplication by the reciprocal of the
 * denominator
 */
Nodeptr Node::divide(const Nodeptr &n1, double rhs)
{
    Nodeptr out = create(n1->data / rhs, "/");
    out->prev = {n1};
    out->former = [n1_weak = std::weak_ptr<Node>(n1),
                   out_weak = std::weak_ptr<Node>(out),
                   rhs]()
    {
        double reciprocal = pow(rhs, -1);
        n1_weak.lock()->gradient = n1_weak.lock()->gradient + (out_weak.lock()->gradient * reciprocal);
    };
    return out;
}

/**
 * SUM OPERATION: C = sum(A)  (reduces entire matrix to scalar)
 *
 * Forward: Sums all elements in matrix A into a single value
 *   C = ΣᵢΣⱼ A[i,j]  (result is 1×1 matrix)
 *
 * Backward: Gradient distributes equally to all input elements
 *   ∂C/∂A[i,j] = 1  (for every element)
 *   So: grad_A[i,j] = grad_C  (same value for all positions)
 *
 * Why? Each input element contributes equally to the sum.
 * Increasing any A[i,j] by δ increases C by exactly δ.
 *
 * USE CASES:
 *   - Reducing loss to scalar (MSE, BCE)
 *   - Sum of squares regularization
 *   - Batch processing (sum over batch dimension)
 */
Nodeptr Node::sum(const Nodeptr &C)
{
    Nodeptr out = create(Matrix(1, 1, {C->data.sumValues()}), "SUM");
    out->prev = {C};
    out->former = [C1_weak = std::weak_ptr<Node>(C),
                   out_weak = std::weak_ptr<Node>(out)]()
    {
        auto Cptr = C1_weak.lock();
        auto outptr = out_weak.lock();

        if (!Cptr || !outptr)
            return;

        Matrix ones(Cptr->data.getRows(), Cptr->data.getCols());
        ones.fill(1.0);

        C1_weak.lock()->gradient = C1_weak.lock()->gradient + (ones * out_weak.lock()->gradient);
    };
    return out;
}

void Node::backprop()
{
    // Stack represents topologically sorted nodes (from last operation to first)
    // Can represent opposite if reversed
    std::stack<Nodeptr> topoStk;
    std::unordered_set<Nodeptr> visited;

    // Get linear ordering of nodes in our graph which respects dependencies

    /** ALSO --> This is why we inherit form enable_shared_from_this,
        we can return the current shared_ptr without creating a copy
     */
    Node::constructTopo(shared_from_this(), topoStk, visited);

    this->gradient.fill(1); // We initialize loss gradient with 1 at start

    // Traverse through stack, and propagate gradients
    while (!topoStk.empty())
    {
        Nodeptr node = topoStk.top();

        if (node->former)
        {
            node->former();
        }
        topoStk.pop();
    }
}

/**  This is a critical component of back propagation as we construct the DAG
    (Directed Acyclic Graph) for our nodes, representing the hierachies. And
    importantly how our gradients should flow downwards*/
void Node::constructTopo(std::shared_ptr<Node> n,
                         std::stack<Nodeptr> &topoStk,
                         std::unordered_set<Nodeptr> &visited)
{

    visited.insert(n);
    for (const auto &child : n->prev)
    {
        // Check if we haven't visited this node already
        if (visited.find(child) == visited.end())
            constructTopo(child, topoStk, visited);
    }

    topoStk.push(n);
}

Nodeptr Node::relu(const Nodeptr &input)
{
    Nodeptr out = create(Matrix::relu(input->data), "ReLu");
    out->prev = {input};
    out->former = [input_weak = std::weak_ptr<Node>(input),
                   out_weak = std::weak_ptr<Node>(out)]()
    {
        auto input = input_weak.lock();
        auto out = out_weak.lock();

        input->gradient = input->gradient +
                          Matrix::hadamardProd(Matrix::relu_derivative(input->data), out->gradient);
    };
    return out;
}

Nodeptr Node::tanh(const Nodeptr &input)
{
    Nodeptr out = create(Matrix::tanh(input->data), "tanh");
    out->prev = {input};
    out->former = [input_weak = std::weak_ptr<Node>(input),
                   out_weak = std::weak_ptr<Node>(out)]()
    {
        auto input = input_weak.lock();
        auto out = out_weak.lock();

        input->gradient = input->gradient +
                          Matrix::hadamardProd(Matrix::tanh_derivative(input->data), out->gradient);
    };
    return out;
}

Nodeptr Node::sigmoid(const Nodeptr &input)
{
    Nodeptr out = create(Matrix::sigmoid(input->data), "sigmoid");
    out->prev = {input};
    out->former = [input_weak = std::weak_ptr<Node>(input),
                   out_weak = std::weak_ptr<Node>(out)]()
    {
        auto input = input_weak.lock();
        auto out = out_weak.lock();
        // input->data.toString();
        // out->gradient.toString();

        input->gradient = input->gradient +
                          Matrix::hadamardProd(Matrix::sigmoid_derivative(input->data), out->gradient);
    };
    return out;
}

Nodeptr Node::leaky_relu(const Nodeptr &input)
{
    Nodeptr out = create(Matrix::leaky_relu(input->data), "leaky ReLu");
    out->prev = {input};
    out->former = [input_weak = std::weak_ptr<Node>(input),
                   out_weak = std::weak_ptr<Node>(out)]()
    {
        auto input = input_weak.lock();
        auto out = out_weak.lock();

        input->gradient = input->gradient +
                          Matrix::hadamardProd(Matrix::leaky_relu_derivative(input->data), out->gradient);
    };
    return out;
}

void Node::generatePseudoRand(double start, double end)
{
    this->data.generatePseudoRand(start, end);
}

void Node::zeroGradient()
{
    this->gradient.fill(0);
}

/**
 * MEAN SQUARED ERROR: L = (y_pred - y_true)² / n
 *
 * Forward: Average of squared differences
 * Backward:
 *   ∂L/∂y_pred = 2*(y_pred - y_true) / n
 *
 * Use for: Regression problems (predicting numbers)
 * Property: Penalizes large errors more severely
 */
Nodeptr Node::mse(const Nodeptr &y_true, const Nodeptr &y_pred)
{
    Nodeptr diff = Node::subtract(y_pred, y_true);
    Nodeptr sq = Node::power(diff, 2);
    Nodeptr sum = Node::sum(sq);
    Nodeptr loss = Node::divide(sum,
                                (double)y_true->data.getSize());

    return loss;
}

/**
 * BINARY CROSS ENTROPY LOSS: L = -[y*log(p) + (1-y)*log(1-p)]
 *
 * Forward: Measures difference between probabilities
 * Backward:
 *   ∂L/∂p = (p - y) / (p*(1-p))
 *   But with sigmoid, this simplifies to (p - y)
 *
 * Use for: Binary classification (XOR problem)
 * Property: Works better than MSE for probabilities
 */
Nodeptr Node::bce(const Nodeptr &y_true, const Nodeptr &y_pred)
{
    const double eps = 1e-8;

    size_t rows = y_true->data.getRows();
    size_t cols = y_true->data.getCols();
    size_t N = rows * cols;

    double lossSum = 0.0;

    // -------- forward pass --------
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            double y = y_true->data.get(i, j);
            double a = y_pred->data.get(i, j);

            // Clamp bad predictions to prevent undefined behavior with logarithims
            // i.e. if a is 0, then clamp to epsilon value (really samll but not 0)
            //      if a is 1, then clamp to 1 - eps (close but not 1)
            //      if anything else then keep as is
            a = std::clamp(a, eps, 1 - eps);

            lossSum += -(y * log(a) + (1.0 - y) * log(1.0 - a));
        }
    }

    lossSum /= N;

    Nodeptr out = create(Matrix(1, 1, {lossSum}), "BCE");
    out->prev = {y_true, y_pred};

    // -------- backward pass --------
    out->former = [y_true_weak = std::weak_ptr<Node>(y_true),
                   y_pred_weak = std::weak_ptr<Node>(y_pred),
                   rows, cols, N]()
    {
        auto y_true = y_true_weak.lock();
        auto y_pred = y_pred_weak.lock();

        if (!y_true || !y_pred)
            return;

        Matrix grad(rows, cols);

        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                double y = y_true->data.get(i, j);
                double a = y_pred->data.get(i, j);

                // BCE + sigmoid simplification:
                // dL/da = (a - y) / N
                double g = (a - y) / N;

                grad.set(i, j, g);
            }
        }

        y_pred->gradient = y_pred->gradient + grad;
    };

    return out;
}

/**
 * SOFTMAX CROSS ENTROPY LOSS
 *
 * Assumes:
 *   - y_true: One-hot encoded matrix (batch × num_classes)
 *   - y_pred: Raw logits (before softmax) OR probabilities
 *
 * Usually used with softmax activation on output layer.
 * ***Currently will only work with softmax activation***
 */
Nodeptr Node::softmaxCE(const Nodeptr &y_true, const Nodeptr &y_pred)
{

    Matrix probabilities = softmax(y_pred);
    // probabilities.toString();

    const double eps = 1e-8;

    // std::cout << probabilities.getSize() << std::endl;
    // std::cout << y_true->data.getSize() << std::endl;

    size_t rows = y_true->data.getRows();
    size_t cols = y_true->data.getCols();
    size_t batch_size = rows;

    double lossSum = 0.0;

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            double y = y_true->data.get(i, j);
            double a = probabilities.get(i, j);

            if (y == 1.0) // Check intended encoding (the correct predicition)
                          // e.g. [0 1 0 0] <-- We ignore the 0s here and check
                          // the models prediction at that position
            {
                //-[y*log(p) + (1-y)*log(1-p)]
                // Simplifies to -log(p) since we know y = 1
                double p = std::clamp(a, eps, 1.0 - eps);
                lossSum += -log(p);
                break;
            }
        }
    }

    lossSum /= batch_size;

    Nodeptr out = create(Matrix(1, 1, {lossSum}), "SoftmaxCE");
    out->prev = {y_true, y_pred};

    out->former = [y_true_weak = std::weak_ptr(y_true),
                   y_pred_weak = std::weak_ptr(y_pred),
                   probabilities,
                   rows, cols, batch_size]()
    {
        auto y_true = y_true_weak.lock();
        auto y_pred = y_pred_weak.lock();

        if (!y_true || !y_pred)
            return;

        Matrix grad(rows, cols);

        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                double y = y_true->data.get(i, j);
                double a = probabilities.get(i, j);

                // CE + softmax simplification:
                // dL/da = (a - y) / batch_size
                double g = (a - y) / batch_size;

                grad.set(i, j, g);
            }
        }

        y_pred->gradient = y_pred->gradient + grad;
    };

    return out;
}

/**
 * SOFTMAX: Converts raw scores to probabilities
 *
 * Formula: p_i = e^z_i / Σ e^z_j
 *
 * Input: Raw scores (logits) for 3 classes
 * Output: Probabilities that sum to 1
 */
Matrix Node::softmax(const Nodeptr &logitsNode)

{
    size_t batch_size = logitsNode->getData().getRows();
    size_t features = logitsNode->getData().getCols();

    Matrix logits = logitsNode->getData();
    Matrix probabilities(batch_size, features);

    for (size_t i = 0; i < batch_size; i++)
    {

        // 1. Find max_value in batch e.g. 3 in [1, 2, 0, 3]
        double max_val = logits.get(i, 0);
        for (size_t j = 0; j < features; j++)
        {
            max_val = std::max(max_val, logits.get(i, j));
        }

        // 2. Find exponentials and calculate exponential sum
        double running_sum_exp = 0.0;
        for (size_t j = 0; j < features; j++)
        {
            // Subtracting max_val yields same result but is safer (does not overflow)
            double exp_val = exp(logits.get(i, j) - max_val);
            probabilities.set(i, j, exp_val);
            running_sum_exp += exp_val;
        }

        // 3. Divide each e^z_i in batch by the running_sum_exp for layer
        for (size_t j = 0; j < features; j++)
        {
            probabilities.set(i, j, probabilities.get(i, j) / running_sum_exp);
        }
    }

    return probabilities;
}

void Node::toString() const
{
    std::cout << "\nData: " << std::flush;
    this->data.toString();
    std::cout << "\nOperation: " << this->operation << std::endl;
    std::cout << "Gradient " << std::flush;
    this->gradient.toString();
    std::cout << "\n"
              << std::endl;
}
