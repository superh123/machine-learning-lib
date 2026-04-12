#pragma once
#include <iostream>
#include <vector>
#include <initializer_list>
#include <stdexcept>

class Matrix
{
public:
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, const std::initializer_list<double> &list);
    Matrix() {}

    Matrix &operator=(const Matrix &other); // Copy assignment
    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(const Matrix &other);
    Matrix operator^(double power); // Element wise exponentiation
    Matrix operator/(double power); // Element wise division

    static Matrix transpose(const Matrix &mat);
    static Matrix identity(const Matrix &mat);

    void fill(double x);      // Fill current matrix
    void scale(double scale); // Scale current matrix values by scaling factor
    void broadcast(int rows, int cols);

    double sumValues();

    double get(int r, int c) const { return this->data.at(r * this->cols + c); }
    void set(int r, int c, double val) { this->data.at(r * this->cols + c) = val; }

    void toString();

private:
    std::vector<double> data;
    int rows;
    int cols;
};
