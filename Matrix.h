#pragma once
#include <iostream>
#include <vector>
#include <initializer_list>
#include <stdexcept>

/**
 * Matrix class for all neural network data
 * ~2D Tensor~ more formally
 *
 * Used for:
 * - Training data (samples × input_features)
 * - Weight matrices (input_features × output_neurons)
 * - Bias vectors (1 × output_neurons)
 * - Predictions (samples × output_neurons)
 * - Layer activations
 */
class Matrix
{
public:
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, const std::initializer_list<double> &list);
    Matrix(size_t rows, size_t cols, double val);
    Matrix() {}

    Matrix(const Matrix &) = default;
    Matrix(Matrix &&) = default;

    /** Assignment operator, reassigns current matrix object with other*/
    Matrix &operator=(const Matrix &other);

    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;

    /**Strictly matrix multiplication OR broadcasting if needed for convenience sake (1x1 matrix) */
    Matrix operator*(const Matrix &other);

    /**Broadcasted multiplication of matrix with scalar*/
    Matrix operator*(double scalar);

    Matrix operator^(double power);
    Matrix operator/(double power);

    static Matrix hadamardProd(const Matrix &m1, const Matrix &m2);

    /**Activation functions and their derivatives*/
    static Matrix relu(const Matrix &input);
    static Matrix relu_derivative(const Matrix &input);
    static Matrix tanh(const Matrix &input);
    static Matrix tanh_derivative(const Matrix &input);
    static Matrix sigmoid(const Matrix &input);
    static Matrix sigmoid_derivative(const Matrix &input);
    static Matrix leaky_relu(const Matrix &input);
    static Matrix leaky_relu_derivative(const Matrix &input);

    static Matrix transpose(const Matrix &mat);
    static Matrix identity(const Matrix &mat);

    static Matrix one_hot_encode(const std::vector<int> &class_indices, size_t num_classes);

    /**Used to seed weight matrix data with random values in range (start, end) to break symmetry*/
    void generatePseudoRand(double start, double end);

    void fill(double x);
    void scale(double scale);
    double sumValues();

    /** Creates a new matrix with columnwise sums of parameter
     *  E.g.
     *  mat = [1 1  -- > return [1, 3]
     *         0 2]
     */
    static Matrix columnWiseSum(Matrix &mat);

    /** Return size 2 vector of dimensions of matrix, (rows, cols) */
    std::vector<size_t> getDimensions();

    double get(int r, int c) const { return this->data.at(r * this->cols + c); }
    void set(int r, int c, double val) { this->data.at(r * this->cols + c) = val; }

    size_t getRows() { return this->rows; };
    size_t getCols() { return this->cols; };

    void toString() const;

    double mean() const;

    size_t getSize() { return data.size(); }

private:
    std::vector<double> data;
    size_t rows;
    size_t cols;
};
