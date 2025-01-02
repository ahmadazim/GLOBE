#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <Eigen/Dense>

using VectorXf = Eigen::Matrix<float, Eigen::Dynamic, 1>;

/** 
 * \brief Save an Eigen vector to a binary file 
 */
void saveVector(const VectorXf& vector, const std::string& filename);

#endif