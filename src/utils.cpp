#include "utils.hpp"
#include <fstream>
#include <iostream>

void saveVector(const VectorXf& vector, const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[saveVector] Error opening file: " << filename << std::endl;
        return;
    }
    int size = vector.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(int));
    file.write(reinterpret_cast<const char*>(vector.data()), size * sizeof(float));
    file.close();
    if (!file) {
        std::cerr << "[saveVector] Error writing file: " << filename << std::endl;
    } else {
        std::cout << "[saveVector] Vector saved: " << filename << std::endl;
    }
}
