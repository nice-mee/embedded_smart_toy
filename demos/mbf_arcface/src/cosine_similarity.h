#include <iostream>
#include <array>
#include <cmath>

// Function to compute the dot product of two std::array objects
template <std::size_t N>
float dotProduct(const std::array<float, N>& arrA, const std::array<float, N>& arrB) {
    float product = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        product += arrA[i] * arrB[i];
    }
    return product;
}

// Function to compute the magnitude of a std::array
template <std::size_t N>
float magnitude(const std::array<float, N>& arr) {
    float sum = 0.0;
    for (float val : arr) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

// Function to compute the cosine similarity between two std::array objects
template <std::size_t N>
float cosineSimilarity(const std::array<float, N>& arrA, const std::array<float, N>& arrB) {
    return dotProduct(arrA, arrB) / (magnitude(arrA) * magnitude(arrB));
}