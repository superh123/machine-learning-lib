#include "Matrix.h"
#include <stdexcept>
#include <cmath>

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols)
{
}

Matrix::Matrix(int rows, int cols, const std::initializer_list<double> &list)
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

    for (int i = 0; i < (other.rows * other.cols); i++)
    {
        this->data.at(i) = other.data.at(i);
    }

    return *this;
}

Matrix Matrix::operator+(const Matrix &other) const
{
    // Create result matrix
    Matrix out(this->rows, this->cols);

    if (this->rows != other.rows || this->cols != other.cols)
    {
        throw std::runtime_error("Dimensions of matrices do not match, addition failed");
    }

    // Fill 'out' matrix
    for (int i = 0; i < (this->rows * this->cols); i++)
    {
        out.data.at(i) = this->data.at(i) + other.data.at(i);
    }

    return out;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    /** Subtraction only valid if rows AND cols of current matrix and other matrix align */
    if (this->rows != other.rows && this->rows != other.cols)
    {
        throw std::runtime_error("Dimensions of matrices do not match, subtraction failed");
    }

    Matrix out(rows, cols);

    for (int i = 0; i < (this->rows * this->cols); i++)
    {
        out.data.at(i) = this->data.at(i) - other.data.at(i);
    }

    return out;
}

Matrix Matrix::operator*(const Matrix &other)
{
    /** A (m x n),
        B (n x p)
        AB = (m x p)
    */
    int M = this->rows;
    int N = this->cols;
    int P = other.cols;

    if (other.rows == 1 && other.cols == 1)
    {
        // std::cout << "\nPART1" << std::endl;
        int val = other.get(0, 0);
        Matrix out(this->rows, this->cols);
        std::cout << "\nVAL\n"
                  << val << std::endl;

        for (int i = 0; i < M * N; i++)
        {
            out.data.at(i) = val;
        }

        return out;
    }
    else
    {
        // std::cout << "\nPART2" << std::endl;
        if (this->cols != other.rows)
        {
            throw std::runtime_error("Columns of Matrix A do not match rows of Matrix B, multiplication failed");
        }

        Matrix out(M, P);

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < P; j++)
            {
                for (int k = 0; k < N; k++)
                {
                    out.data.at(i * other.cols + j) += this->get(i, k) * other.get(k, j);
                }
            }
        }

        return out;
    }
}

Matrix Matrix::operator^(double power)
{
    Matrix out(this->rows, this->cols);

    for (int i = 0; i < rows * cols; i++)
    {
        out.data.at(i) = pow(this->data.at(i), power);
    }

    return out;
}

Matrix Matrix::operator/(double power)
{
    Matrix out(this->rows, this->cols);

    for (int i = 0; i < rows * cols; i++)
    {
        out.data.at(i) = this->data.at(i) / power;
    }

    return out;
}

Matrix Matrix::transpose(const Matrix &mat)
{
    // If A is (m X n), A^T is (n x m)
    Matrix transpose(mat.cols, mat.rows);

    int M = mat.rows;
    int N = mat.cols;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            transpose.set(j, i, mat.get(i, j));
        }
    }

    // std::cout << "Transposing" << std::endl;
    // transpose.toString();
    std::cout << "\n"
              << std::endl;

    return transpose;
}

void Matrix::fill(double x)
{
    for (int i = 0; i < rows * cols; i++)
    {
        this->data.at(i) = x;
    }
}

Matrix Matrix::identity(const Matrix &mat)
{
    Matrix identity = Matrix(mat.rows, mat.cols);
    identity.fill(0); // Identity matrix

    int rows = mat.rows;
    int col = 0;

    for (int i = 0; i < rows; i++)
    {
        identity.set(i, col++, 1);
    }

    return identity;
}

void Matrix::scale(double scale)
{
    for (int i = 0; i < rows * cols; i++)
    {
        this->data.at(i) *= scale;
    }
}

double Matrix::sumValues()
{
    double sum = 0;
    for (int i = 0; i < this->data.size(); i++)
    {
        sum += this->data.at(i);
    }

    return sum;
}

//
void Matrix::toString()
{
    for (int i = 0; i < this->data.size(); i++)
    {
        std::cout << this->data.at(i) << " " << std::flush;
    }
}
