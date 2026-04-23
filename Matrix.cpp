#include "Matrix.h"
#include <stdexcept>
#include <cmath>
#include <random>
#include <iomanip>

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(rows * cols)
{
}

/** Constructor to initialize Matrix object with given data */
Matrix::Matrix(size_t rows, size_t cols, const std::initializer_list<double> &list)
    : rows(rows), cols(cols), data(list)
{
    if (list.size() != (rows * cols))
    {
        throw std::runtime_error("On initialization: List size does not match matrix dimensions");
    }
}

Matrix &Matrix::operator=(const Matrix &other)
{

    this->data.resize(other.data.size());
    this->rows = other.rows;
    this->cols = other.cols;

    for (size_t i = 0; i < (other.rows * other.cols); i++)
    {
        this->data.at(i) = other.data.at(i);
    }

    return *this;
}

Matrix Matrix::operator+(const Matrix &other) const
{
    // Create result matrix
    Matrix out(this->rows, this->cols);

    if (other.rows == 1 && this->cols == other.cols) // Check if broadcasting needed instead? (Used for bias addition)
    {
        for (size_t c = 0; c < this->cols; c++)
        {
            for (size_t r = 0; r < (this->rows); r++)
            {
                out.set(r, c, other.get(0, c) + this->get(r, c));
            }
        }

        return out;
    }
    else if (this->rows == 1 && other.cols == this->cols) // Check if broadcasting needed instead? (Used for bias addition)
    {
        for (size_t c = 0; c < other.cols; c++)
        {
            for (size_t r = 0; r < (other.rows); r++)
            {
                out.set(r, c, this->get(0, c) + other.get(r, c));
            }
        }

        return out;
    }

    // If we reach here, we know we are doing traditional matrix addition (element-wise)

    if (this->rows != other.rows || this->cols != other.cols)
    {
        throw std::runtime_error("Dimensions of matrices do not match, addition failed");
    }

    for (size_t i = 0; i < (this->rows * this->cols); i++)
    {
        out.data.at(i) = this->data.at(i) + other.data.at(i);
    }

    return out;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    // Subtraction only valid if rows AND cols of current matrix and other matrix align
    // NOTE: ADD BROADCASTING?
    if (this->rows != other.rows && this->rows != other.cols)
    {
        throw std::runtime_error("Dimensions of matrices do not match, subtraction failed");
    }

    Matrix out(rows, cols);

    for (size_t i = 0; i < (this->rows * this->cols); i++)
    {
        out.data.at(i) = this->data.at(i) - other.data.at(i);
    }

    return out;
}

Matrix Matrix::operator*(const Matrix &other)
{
    size_t M = this->rows;
    size_t N = this->cols;
    size_t P = other.cols;

    if (other.rows == 1 && other.cols == 1)
    {
        // Broadcast multiplication elementwise
        // FIXME: PERHAPS NOT NEEDED? SEE OVERLOADED OPERATOR, UPDATE LINGERING DEPENDENCIES

        double val = other.get(0, 0);
        Matrix out(this->rows, this->cols);

        for (size_t i = 0; i < M * N; i++)
        {
            out.data.at(i) = this->data.at(i) * val;
        }

        return out;
    }
    else
    {
        // Matrix multiplication needed if we reach here

        /** A (m x n)
            B (n x p)
            AB = (m x p)
        */

        if (this->cols != other.rows)
        {
            throw std::runtime_error("Columns of Matrix A do not match rows of Matrix B, multiplication failed");
        }

        Matrix out(M, P);

        for (size_t i = 0; i < M; i++)
        {
            for (size_t j = 0; j < P; j++)
            {
                double sum = 0.0;

                for (size_t k = 0; k < N; k++)
                {
                    sum += this->get(i, k) * other.get(k, j);
                }

                out.data.at(i * P + j) = sum;
            }
        }

        return out;
    }
}

Matrix Matrix::operator*(double scalar)
{
    Matrix out = *this;

    out.scale(scalar);

    return out;
}

Matrix Matrix::hadamardProd(const Matrix &m1, const Matrix &m2)
{

    // Used for elementwise multiplication of matrices

    Matrix out(m1.rows, m1.cols);

    if (m1.rows != m2.rows || m1.cols != m2.cols)
    {
        throw std::runtime_error("Dimensions of matrices do not match, addition failed");
    }

    for (size_t i = 0; i < (m1.rows * m1.cols); i++)
    {
        out.data.at(i) = m1.data.at(i) * m2.data.at(i);
    }

    return out;
}

Matrix Matrix::operator^(double power)
{
    Matrix out(this->rows, this->cols);

    for (size_t i = 0; i < rows * cols; i++)
    {
        out.data.at(i) = pow(this->data.at(i), power);
    }

    return out;
}

Matrix Matrix::operator/(double power)
{
    Matrix out(this->rows, this->cols);

    for (size_t i = 0; i < rows * cols; i++)
    {
        out.data.at(i) = this->data.at(i) / power;
    }

    return out;
}

Matrix Matrix::relu(const Matrix &input)
{
    // Formula : f(x) = max(0, x);

    Matrix out(input.rows, input.cols);

    for (size_t i = 0; i < input.data.size(); i++)
    {
        out.data.at(i) = std::max(0.0, input.data.at(i));
    }

    return out;
}

Matrix Matrix::relu_derivative(const Matrix &input)
{

    // Formula : f'(x) = 1 if x > 0 OR f'(x) = 0 if x <= 0

    Matrix out(input.rows, input.cols);

    for (size_t i = 0; i < input.data.size(); i++)
    {
        out.data.at(i) = (input.data.at(i) > 0 ? 1 : 0);
    }

    return out;
}

Matrix Matrix::tanh(const Matrix &input)
{
    // Formula : f(x) = sinh/cosh

    Matrix out = Matrix(input.rows, input.cols);

    for (size_t i = 0; i < input.data.size(); i++)
    {
        out.data.at(i) = std::sinh(input.data.at(i)) / (double)std::cosh(input.data.at(i));
    }
    return out;
}

Matrix Matrix::tanh_derivative(const Matrix &input)
{
    // Formula : f'(x) = sech^2(x) = 1 - tanh^2(x)

    Matrix out = Matrix(input.rows, input.cols);

    for (size_t i = 0; i < input.data.size(); i++)
    {
        // std::cout << 1 - pow(std::tanh(input.data.at(i)), 2) << std::endl;
        out.data.at(i) = 1 - pow(std::tanh(input.data.at(i)), 2);
    }

    return out;
}

Matrix Matrix::sigmoid(const Matrix &input)
{
    // Formula : f(x) = 1 / (1 + e^-x)

    Matrix out(input.rows, input.cols);

    for (size_t i = 0; i < input.data.size(); i++)
    {
        double x = input.data.at(i);
        double e = exp(1);
        double sigmoid_x = 1 / (1 + pow(e, -x));
        out.data.at(i) = sigmoid_x;
    }

    return out;
}

Matrix Matrix::sigmoid_derivative(const Matrix &input)
{
    // Formula : f'(x) = f(x) * (1 - f(x))

    Matrix out(input.rows, input.cols);

    for (size_t i = 0; i < input.data.size(); i++)
    {
        double x = input.data.at(i);
        double e = exp(1);
        double sigmoid_x = 1 / (1 + pow(e, -x));
        out.data.at(i) = sigmoid_x * (1 - sigmoid_x);
    }

    return out;
}

Matrix Matrix::leaky_relu(const Matrix &input)
{
    // Formula : f(x) = max(ax, x) where a is a small constant

    Matrix out(input.rows, input.cols);

    double constant = 0.01;

    for (size_t i = 0; i < input.data.size(); i++)
    {
        out.data.at(i) = std::max(constant * input.data.at(i), input.data.at(i));
    }

    return out;
}

Matrix Matrix::leaky_relu_derivative(const Matrix &input)
{

    // Formula : f'(x) = 1 if x > 0 OR f'(x) = a if x <= 0 (where a is a small constant)

    Matrix out(input.rows, input.cols);

    double constant = 0.01;

    for (size_t i = 0; i < input.data.size(); i++)
    {
        out.data.at(i) = (input.data.at(i) > 0 ? 1 : constant);
    }

    return out;
}

Matrix Matrix::transpose(const Matrix &mat)
{
    // If A is (m X n), A^T is (n x m)
    // E.g. A element at (0, 1) in the matrix
    // is placed at (1, 0) in the transpose

    Matrix transpose(mat.cols, mat.rows);

    size_t M = mat.rows;
    size_t N = mat.cols;

    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            transpose.set(j, i, mat.get(i, j));
        }
    }

    return transpose;
}

void Matrix::fill(double x)
{
    for (size_t i = 0; i < rows * cols; i++)
    {
        this->data.at(i) = x;
    }
}

Matrix Matrix::identity(const Matrix &mat)
{
    Matrix identity = Matrix(mat.rows, mat.cols);
    identity.fill(0);

    size_t rows = mat.rows;
    size_t col = 0;

    for (size_t i = 0; i < rows; i++)
    {
        identity.set(i, col++, 1);
    }

    return identity;
}

void Matrix::generatePseudoRand(double start, double end)
{
    // Explained in header

    std::random_device rd;

    std::mt19937 gen(rd());

    std::uniform_real_distribution<> distrib(start, end);

    for (size_t i = 0; i < this->data.size(); i++)
    {
        this->data.at(i) = distrib(gen);
    }

    return;
}

/**
 * Convert class indices to one-hot vectors
 *
 * @param class_indices Vector of class labels (0, 1, 2, ...)
 * @param num_classes Total number of classes
 * @return Matrix where each row is one-hot encoded
 */
Matrix Matrix::one_hot_encode(const std::vector<int> &labels, size_t num_classes)
{
    size_t num_samples = labels.size();

    Matrix one_hot(num_samples, num_classes);

    one_hot.fill(0);

    for (size_t i = 0; i < num_samples; i++)
    {
        int class_idx = labels[i];
        one_hot.set(i, class_idx, 1.0);
    }

    return one_hot;
}

void Matrix::scale(double scale)
{
    for (size_t i = 0; i < rows * cols; i++)
    {
        this->data.at(i) *= scale;
    }
}

double Matrix::sumValues()
{
    double sum = 0;
    for (size_t i = 0; i < this->data.size(); i++)
    {
        sum += this->data.at(i);
    }

    // Return scalar sum of all values in matrix
    return sum;
}

Matrix Matrix::columnWiseSum(Matrix &mat)
{
    // Explained in header

    Matrix out = Matrix(1, mat.cols);

    for (size_t c = 0; c < mat.cols; c++)
    {
        double sum = 0;
        for (size_t r = 0; r < (mat.rows); r++)
        {
            sum += mat.get(r, c);
        }
        out.set(0, c, sum);
    }

    return out;
}

std::vector<size_t> Matrix::getDimensions()
{
    std::vector<size_t> dim{rows, cols};
    return dim;
}

double Matrix::mean() const
{
    double sum = 0.0;
    size_t count = rows * cols;

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            sum += this->get(i, j);
        }
    }

    return sum / count;
}

void Matrix::toString() const
{
    for (size_t i = 0; i < this->data.size(); i++)
    {
        std::cout << this->data.at(i) << " " << std::flush;
        if (i != 0 && i % (cols - 1) == 0)
            std::cout << "\n";
    }
}
