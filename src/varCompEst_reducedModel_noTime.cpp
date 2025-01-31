#include "utils.hpp" 
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unordered_map>
#include <omp.h>
#include <cassert>
#include <cmath>
#include <unordered_set>
#include <random>
#include <streambuf>

#include <nlopt.hpp>

// Global constants
bool DEBUG = true;
int MAX_ITER = 50;
float TOL = 1e-3;
int NO_MC_SAMPLES = 15; 

// Typedefs
using namespace Eigen;
using MatrixXf = Matrix<float, Dynamic, Dynamic>;
using VectorXf = Matrix<float, Dynamic, 1>;

// Struct definitions
struct CohortDataStruct {
    VectorXf indiv, A_vecLT, Y;
    MatrixXf X;
    int N, p;
};

struct SolvingDataStruct {
    VectorXf Vinv_r, Vinv_Y, beta_hat, u_hat;
    MatrixXf Vinv_X, Xt_Vinv_X_inv;
    VectorXf Qdiag_inv;
};

struct VarCompDataStruct {
    float sig2e = 0.5f;
    float sig2g = 0.5f; 
    const int num_varComps = 2;

    void updateFromVector(const VectorXf &theta) {
        sig2e = theta(0);
        sig2g = theta(1);
    }
};

struct TrustRegionData {
    VectorXf gradient;
    MatrixXf AI;
    float Delta;     
    VectorXf theta_prev;
};

// ========================================
//             INPUT PROCESSING            
// ========================================

class TeeStreambuf : public std::streambuf {
public:
    TeeStreambuf(std::streambuf* sb1, std::streambuf* sb2)
        : sb1(sb1), sb2(sb2) {}

protected:
    virtual int overflow(int c) override {
        if (c == EOF) {
            return !EOF;
        } else {
            int const r1 = sb1->sputc(c);
            int const r2 = sb2->sputc(c);
            return (r1 == EOF || r2 == EOF) ? EOF : c;
        }
    }

    virtual int sync() override {
        int const r1 = sb1->pubsync();
        int const r2 = sb2->pubsync();
        return (r1 == 0 && r2 == 0) ? 0 : -1;
    }

private:
    std::streambuf* sb1;
    std::streambuf* sb2;
};


void parseCommandLineArgs(int argc, char* argv[],
                          std::string &input_filename,
                          std::string &grm_prefix,
                          std::string &output_prefix,
                          VarCompDataStruct &VarCompData)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_filename> <grm_prefix> <output_prefix>"
                  << " [--maxiter <int>] [--tol <float>] [--numMC <int>]"
                  << " [--sig2e <float>] [--sig2g <float>]"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    input_filename = argv[1];
    grm_prefix     = argv[2];
    output_prefix  = argv[3];

    int i = 4;
    while (i < argc) {
        std::string arg = argv[i];
        if (arg == "--maxiter") {
            MAX_ITER = std::stoi(argv[++i]);
        } else if (arg == "--tol") {
            TOL = std::stof(argv[++i]);
        } else if (arg == "--numMC") {
            NO_MC_SAMPLES = std::stoi(argv[++i]);
        } else if (arg == "--sig2e") {
            VarCompData.sig2e = std::stof(argv[++i]);
        } else if (arg == "--sig2g") {
            VarCompData.sig2g = std::stof(argv[++i]);
        }
        ++i;
    }
}

// Helper function to check the input file and get N, Nrec, and p
void checkInputs(CohortDataStruct& CohortData, const std::string& filename) {
    int& N = CohortData.N;
    int& p = CohortData.p;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening input file." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    std::getline(file, line); // Read header line
    std::istringstream header(line);
    std::string column;
    std::unordered_set<int> unique_indivs;
    int indivIndex = -1, phenoIndex = -1;
    int index = 0;

    while (header >> column) {
        if (column == "INDIV") {
            indivIndex = index;
        } else if (column == "PHENO") {
            phenoIndex = index;
        }
        ++index;
    }

    if (indivIndex == -1 || phenoIndex == -1) {
        std::cerr << "Error: Missing required columns in header." << std::endl;
        exit(EXIT_FAILURE);
    }

    p = index - 2; // Number of columns excluding INDIV and PHENO
    int Nrec = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float indiv_id;
        for (int i = 0; i < index; ++i) {
            if (i == indivIndex) {
                iss >> indiv_id;
                unique_indivs.insert(static_cast<int>(indiv_id));
            } else {
                float temp;
                iss >> temp;
            }
        }
        ++Nrec;
    }
    N = unique_indivs.size();
    file.close();

    if (DEBUG) {
        std::cout << 
        "Found N = " << N << " indivuduals, " << std::endl << 
        "      p = " << p << " fixed effects, and" << std::endl << 
        "      Nrec = " << Nrec << " records in the input file." << std::endl;
    }
}

void readInputData(CohortDataStruct& CohortData, const std::string& filename) {
    VectorXf& indiv = CohortData.indiv;
    VectorXf& Y = CohortData.Y;
    MatrixXf& X = CohortData.X;
    int& p = CohortData.p;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening input file." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    std::getline(file, line); // header line
    std::istringstream header(line);
    std::string column;
    int indivIndex = -1, phenoIndex = -1;
    int index = 0;

    while (header >> column) {
        if (column == "INDIV") {
            indivIndex = index;
        } else if (column == "PHENO") {
            phenoIndex = index;
        }
        ++index;
    }
    std::cout << "Columns found: INDIV at " << indivIndex << ", PHENO at " << phenoIndex << std::endl;

    int row = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float pheno, indiv_id;
        
        for (int i = 0; i < index; ++i) {
            if (i == indivIndex) {
                iss >> indiv_id;
            } else if (i == phenoIndex) {
                iss >> pheno;
            } else {
                iss >> X(row, i - 2);   // potential bug here: i - 2 (columns of interest might not come before the rest of the covariates...) TO-DO: FIX
            }
        }
        indiv(row) = indiv_id - 1;      // 0-indexed individual ID
        Y(row) = pheno;
        ++row;
    }
    file.close();

    if (DEBUG) {
        std::cout << "First 5 rows of X:" << std::endl;
        std::cout << X.topRows(5) << std::endl;
        std::cout << "First 5 elements of Y:" << std::endl;
        std::cout << Y.head(5) << std::endl;
        std::cout << "First 5 elements of indiv:" << std::endl;
        std::cout << indiv.head(5) << std::endl;
    }
}

void readGRM(const std::string &grm_prefix, CohortDataStruct &CohortData)
{
    std::string filename = grm_prefix + ".grm.bin";
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening GRM file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    size_t nTri = (size_t)CohortData.N * (CohortData.N + 1) / 2;
    CohortData.A_vecLT.resize(nTri);

    file.read(reinterpret_cast<char*>(CohortData.A_vecLT.data()), nTri * sizeof(float));
    file.close();

    if (DEBUG && CohortData.N > 0) {
        std::cout << "[readGRM] Top-left of A:\n";
        for (int i=0; i < std::min(5, CohortData.N); i++) {
            for (int j=0; j < std::min(5, CohortData.N); j++) {
                int iLow = (i >= j ? i : j);
                int jLow = (i >= j ? j : i);
                size_t indexLT = (size_t)iLow * (iLow + 1)/2 + jLow;
                float val = CohortData.A_vecLT(indexLT);
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
}

// ========================================
//              SOLVING MMEs            
// ========================================

// Helper function to access i, j element of symmetric matrix stored in vector (lower triangular portion)
inline float getElement_vecLT(const VectorXf& vec, size_t i, size_t j) {
    return (i < j) ? vec(j * (j + 1) / 2 + i) : vec(i * (i + 1) / 2 + j);
}

// Function to compute elements of V on the fly
float computeV_ij(const CohortDataStruct& CohortData, const VarCompDataStruct& VarCompData, 
                  int i, int j, bool noGRM = false) {
    float genetic_component = VarCompData.sig2g; 
    genetic_component *= noGRM ? 1.0f : getElement_vecLT(CohortData.A_vecLT, i, j);
    float residual_component = (i == j) ? VarCompData.sig2e : 0.0f;
    return genetic_component + residual_component;
}

void computeVx_multi_VC(const CohortDataStruct& CohortData, const SolvingDataStruct& SolvingData, const VarCompDataStruct& VarCompData,
                     const MatrixXf& input_matrix, MatrixXf& result_matrix) {

    int num_rhs = input_matrix.cols();
    result_matrix.setZero();
    std::vector<MatrixXf> local_results(omp_get_max_threads(), MatrixXf::Zero(CohortData.N, num_rhs));

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        MatrixXf& local_result = local_results[thread_id];  // Each thread has its own local result matrix
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < CohortData.N; ++i) {
            for (int j = 0; j <= i; ++j) {
                float V_ij = computeV_ij(CohortData, VarCompData, i, j);
                for (int k = 0; k < num_rhs; ++k) {
                    local_result(i, k) += V_ij * input_matrix(j, k);
                    if (i != j) {
                        local_result(j, k) += V_ij * input_matrix(i, k);
                    }
                }
            }
        }
        #pragma omp critical
        result_matrix += local_result;
    }
}


void computePreconditioner(const CohortDataStruct& CohortData, SolvingDataStruct& SolvingData, const VarCompDataStruct& VarCompData) {
    int N = CohortData.N;
    SolvingData.Qdiag_inv.resize(N);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        float diagVal = VarCompData.sig2g * getElement_vecLT(CohortData.A_vecLT, i, i) + VarCompData.sig2e;
        SolvingData.Qdiag_inv(i) = 1.0f / diagVal;
    }
}

// Function to run PCG (Preconditioned Conjugate Gradient) algorithm to solve Vx = b for MULTIPLE b simulataneously
void runPCG_multi_VC(const CohortDataStruct& CohortData, const SolvingDataStruct& SolvingData, const VarCompDataStruct& VarCompData,
                  const MatrixXf& b, MatrixXf& x, float tol, int max_iter, bool use_input_initial = false, bool use_initial_guess = false, bool verbose = false) {
    // `use_input_initial` uses initial guess provided in `x`
    // `use_initial_guess` uses initial guess computed from the Q^{-1}b
    
    const int N = CohortData.N; 
    int num_rhs = b.cols();
    
    // x = MatrixXf::Zero(N, num_rhs);
    MatrixXf r(N, num_rhs);
    if (use_input_initial) {
        MatrixXf Vx = MatrixXf::Zero(N, num_rhs);
        computeVx_multi_VC(CohortData, SolvingData, VarCompData, x, Vx);
        r = b - Vx;
    } else if (use_initial_guess) {
        VectorXf bcol(N), xcol(N); 
        for (int j = 0; j < num_rhs; ++j) {
            bcol = b.col(j);
            for (int i = 0; i < N; ++i) {
                xcol(i) = bcol(i) * SolvingData.Qdiag_inv(i);
            }
            x.col(j) = xcol;
        }
        MatrixXf Vx = MatrixXf::Zero(N, num_rhs);
        computeVx_multi_VC(CohortData, SolvingData, VarCompData, x, Vx);
        r = b - Vx;
    } else {
        x = MatrixXf::Zero(N, num_rhs);
        r = b;
    }

    MatrixXf z(N, num_rhs);
    MatrixXf Vp(N, num_rhs);
    MatrixXf p(N, num_rhs);
    
    VectorXf zcol(N), rcol(N); 
    for (int j = 0; j < num_rhs; ++j) {
        rcol = r.col(j);
        for (int i = 0; i < N; ++i) {
            zcol(i) = rcol(i) * SolvingData.Qdiag_inv(i);
        }
        z.col(j) = zcol;
    }
    p = z;
    VectorXf rsold = (r.array() * z.array()).colwise().sum();

    for (int iter = 0; iter < max_iter; ++iter) {
        computeVx_multi_VC(CohortData, SolvingData, VarCompData, p, Vp);
        VectorXf alpha = rsold.array() / (p.array() * Vp.array()).colwise().sum().transpose();
        
        for (int j = 0; j < num_rhs; ++j) {
            x.col(j).noalias() += alpha(j) * p.col(j);
            r.col(j).noalias() -= alpha(j) * Vp.col(j);
        }
        
        // TO-DO: check convergence individually for each RHS and stop that RHS if converged (continue only with others)
        if ((r.colwise().norm().array() < tol * b.colwise().norm().array()).all()) {
            std::cout << "Converged at iteration " << iter + 1 << std::endl;
            break;
        }
        for (int j = 0; j < num_rhs; ++j) {
            rcol = r.col(j);
            for (int i = 0; i < N; ++i) {
                zcol(i) = rcol(i) * SolvingData.Qdiag_inv(i);
            }
            z.col(j) = zcol;
        }
        VectorXf rsnew = (r.array() * z.array()).colwise().sum();
        VectorXf beta = rsnew.array() / rsold.array(); 
        rsold = rsnew; 

        for (int j = 0; j < num_rhs; ++j) {
            p.col(j).noalias() = z.col(j) + beta(j) * p.col(j);
        }

        if (verbose) {
            for (int j = 0; j < num_rhs; ++j) {
                std::cout << "RHS " << j << ", Iteration " << iter + 1 << ", residual norm / b norm: " << r.col(j).norm() / b.col(j).norm() << std::endl;
            }
        }
    }
}

// Function to compute u_hat efficiently using precomputed records for each individual
void compute_uhat_VC(const CohortDataStruct& CohortData, SolvingDataStruct& SolvingData, const VarCompDataStruct& VarCompData) {
    const int N = CohortData.N;
    const VectorXf& A_vecLT = CohortData.A_vecLT;
    const VectorXf& Vinv_r = SolvingData.Vinv_r;
    VectorXf& u_hat = SolvingData.u_hat;
    u_hat.setZero(N);
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            sum += getElement_vecLT(A_vecLT, i, j) * Vinv_r[j];
        }
        u_hat[i] = VarCompData.sig2g * sum;
    }
}

// Function to fit the mixed model equations given variance component params
void fitMME(const CohortDataStruct& CohortData, SolvingDataStruct& SolvingData, const VarCompDataStruct& VarCompData, bool verbose = false) {

    if (DEBUG) std::cout << "Solving MME... ";
    
    const int N = CohortData.N;
    const int p = CohortData.p;
    const MatrixXf& X = CohortData.X;
    const VectorXf& Y = CohortData.Y;
    MatrixXf& Vinv_X = SolvingData.Vinv_X;
    MatrixXf& Xt_Vinv_X_inv = SolvingData.Xt_Vinv_X_inv;
    VectorXf& beta_hat = SolvingData.beta_hat;
    VectorXf& Vinv_r = SolvingData.Vinv_r;
    VectorXf& u_hat = SolvingData.u_hat;
    VectorXf& Vinv_Y = SolvingData.Vinv_Y;


    // #===== STEP 0: Preparation =====#
    
    // Zero out all solving data
    Vinv_X.setZero(N, p);
    Xt_Vinv_X_inv.setZero(p, p);
    beta_hat.setZero(p);
    Vinv_r.setZero(N);
    u_hat.setZero(N);
    Vinv_Y.setZero(N);

    // Compute the block preconditioner Q^{-1}
    computePreconditioner(CohortData, SolvingData, VarCompData);
    if (verbose) {
        std::cout << "First 5 diagonal preconditioner entried:" << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << SolvingData.Qdiag_inv[i] << " ";
        }
        std::cout << std::endl;
    }
    float tol = 1e-5;
    int max_iter = 1000;

    // #===== STEP 1: Fixed effects =====#

    // Step 1a,1b: compute V^{-1}Y
    MatrixXf XY(N, p + 1);
    XY << X, Y;
    MatrixXf Vinv_XY(N, p + 1);
    runPCG_multi_VC(CohortData, SolvingData, VarCompData, XY, Vinv_XY, tol, max_iter, false, true);

    if (verbose) {
        std::cout << "First 5 elements of V^{-1}XY:" << std::endl;
        std::cout << Vinv_XY.topRows(5) << std::endl;
    }
    
    Vinv_X = Vinv_XY.leftCols(p);
    Vinv_Y = Vinv_XY.rightCols(1);

    // Step 1c: compute (X'V^{-1}X)^{-1}
    MatrixXf Xt_Vinv_X = X.transpose() * Vinv_X;
    Xt_Vinv_X_inv = Xt_Vinv_X.inverse();

    // Step 1d: compute \hat{beta}
    beta_hat.noalias() = Xt_Vinv_X_inv * X.transpose() * Vinv_Y; 


    // #===== STEP 2: Random effects =====#
    // Step 2a: compute r = Y - X\hat{beta}
    VectorXf r = Y - X * beta_hat;

    // Step 2b: compute V^{-1}r
    Vinv_r = Vinv_Y - Vinv_X * beta_hat;
    if (verbose) {
        std::cout << "First 5 elements of V^{-1}r:" << std::endl;
        std::cout << Vinv_r.head(5) << std::endl;
    }

    // Step 2c: compute \hat{u}
    compute_uhat_VC(CohortData, SolvingData, VarCompData);
    if (verbose) {
        std::cout << "First 5 elements of u_hat:" << std::endl;
        std::cout << u_hat.head(5) << std::endl;
    }
}


// ==========================================================
//              VARIANCE COMPONENT ESTIMATION            
// ==========================================================

VectorXf compute_f_theta_i(const CohortDataStruct& CohortData, const SolvingDataStruct& SolvingData, const VarCompDataStruct& VarCompData,
                           int theta_index) {
    
    const int N = CohortData.N;
    const VectorXf& Vinv_r = SolvingData.Vinv_r;
    VectorXf f_theta_i(N);

    switch(theta_index) {
        case 0: // SIG2E
        {
            f_theta_i = Vinv_r;
            break;
        }
        case 1: // SIG2G
        {
            for (int i = 0; i < N; ++i) {
                f_theta_i(i) = SolvingData.u_hat(i) / VarCompData.sig2g;
            }
            break;
        }
    }
    return f_theta_i;
}

// Helper function to generate standard normal random variables
VectorXf generate_standard_normal(int dimension) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);
    VectorXf z(dimension);
    for (int i = 0; i < dimension; ++i) {
        z(i) = dist(gen);
    }
    return z;
}

// Helper function to sample matrix of Y_v^* ~ N(0, V^*), V^* = sig2g * I_N + sig2e * I_N
MatrixXf sample_Yvs(const CohortDataStruct& CohortData, const VarCompDataStruct& VarCompData, int num_samples) {

    const int N = CohortData.N;
    MatrixXf Yvs(N, num_samples);

    for (int j = 0; j < num_samples; ++j) {
        VectorXf Z1 = generate_standard_normal(N);
        VectorXf Z3 = generate_standard_normal(N);
        for (int i = 0; i < N; ++i) {
            Yvs(i, j) = sqrt(VarCompData.sig2g) * Z1(i) + sqrt(VarCompData.sig2e) * Z3(i);
        }
    }
    return Yvs;
}

// Function to estimate trace of a matrix (using Monte Carlo sampling)
void estimateTrace_PdV(const CohortDataStruct& CohortData, const VarCompDataStruct& VarCompData, SolvingDataStruct& SolvingData, 
                           int num_samples, VectorXf& trace_estimates, bool verbose = false) {
    
    std::cout << "Estimating trace terms... ";
    const int num_varComps = VarCompData.num_varComps;
    const int N = CohortData.N;
    
    // 1. sample Y_v^* ~ N(0, V^*)
    MatrixXf Yvs = sample_Yvs(CohortData, VarCompData, num_samples);
    if (verbose) {
        std::cout << "First 5 samples of Yvs:" << std::endl;
        std::cout << Yvs.topRows(5) << std::endl;
    }

    // 2. Compute V*^{-1}Y_v (direct block inversion of V* is trivial, so will just do that)
    // std::cout << "Computing tildeVinv_Yvs... ";
    MatrixXf tildeVinv_Yvs(N, num_samples);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        float V_ii = computeV_ij(CohortData, VarCompData, i, i, true);
        float V_ii_inv = 1.0f / V_ii;
        for (int j = 0; j < num_samples; ++j) {
            tildeVinv_Yvs(i, j) = V_ii_inv * Yvs(i, j);
        }
    }
    if (verbose) {
        std::cout << "First 5 samples of tildeVinv_Yvs:" << std::endl;
        std::cout << tildeVinv_Yvs.topRows(5) << std::endl;
    }

    // 3. compute Y_v'P dV/dtheta_i = f(theta_i)' except "Vinv_r" and "u_hat" are different
    // iteratively solve to get V^{-1}Y_vs and then compute PYvs and then compute f(theta_i)s
    // Initalize Vinv_Yvs with tildeVinv_Yv
    MatrixXf Vinv_Yvs(N, num_samples);
    for (int j = 0; j < num_samples; ++j) {
        Vinv_Yvs.col(j) = tildeVinv_Yvs.col(j);
    }
    // std::cout << "Computing Vinv_Yvs... ";
    runPCG_multi_VC(CohortData, SolvingData, VarCompData, Yvs, Vinv_Yvs, 1e-5, 1000, true, false);
    if (verbose) {
        std::cout << "First 5 samples of Vinv_Yvs:" << std::endl;
        std::cout << Vinv_Yvs.topRows(5) << std::endl;
    }
    for (int sample = 0; sample < num_samples; ++sample) {
        VectorXf tildeVinv_Yv = tildeVinv_Yvs.col(sample);
        
        // to use the same functions, we can use same struct and update the Vinv_r and u_hat (no longer used downstream)
        VectorXf Vinv_Yv = Vinv_Yvs.col(sample);

        SolvingData.Vinv_r = Vinv_Yv - SolvingData.Vinv_X * SolvingData.Xt_Vinv_X_inv * (CohortData.X.transpose() * Vinv_Yv);
        compute_uhat_VC(CohortData, SolvingData, VarCompData);
        
        for (int theta_index = 0; theta_index < num_varComps; ++theta_index) {
            VectorXf tilde_f_theta_i = compute_f_theta_i(CohortData, SolvingData, VarCompData, theta_index);
            trace_estimates(theta_index) += tilde_f_theta_i.dot(tildeVinv_Yv);
            if (verbose) std::cout << "trace estimate for theta_" << theta_index << ": " << trace_estimates(theta_index)/(sample+1) << std::endl;
        }
    }
    // Average over samples
    for (int theta_index = 0; theta_index < num_varComps; ++theta_index) {
        trace_estimates(theta_index) /= num_samples;
    }
}

// Compute first derivatives of the log-likelihood function wrt variance components
void computeDerivative_first(const CohortDataStruct& CohortData, const VarCompDataStruct& VarCompData, SolvingDataStruct& SolvingData, 
                             MatrixXf& f_theta, VectorXf& gradient) {
    
    if (DEBUG) std::cout << "Computing derivatives." << std::endl;

    const int num_varComps = VarCompData.num_varComps;
    const int N = CohortData.N;
    
    gradient.resize(num_varComps);
    f_theta.resize(N, num_varComps);

    // Compute f_theta_i for each variance component
    VectorXf first_terms(num_varComps);
    for (int theta_index = 0; theta_index < num_varComps; ++theta_index) {
        VectorXf f_theta_i = compute_f_theta_i(CohortData, SolvingData, VarCompData, theta_index);
        f_theta.col(theta_index) = f_theta_i;
        first_terms(theta_index) = SolvingData.Vinv_r.dot(f_theta_i);
    }

    // Estimate trace terms
    VectorXf trace_estimates = VectorXf::Zero(num_varComps);
    estimateTrace_PdV(CohortData, VarCompData, SolvingData, NO_MC_SAMPLES, trace_estimates);
    
    // Compute gradient
    for (int theta_index = 0; theta_index < num_varComps; ++theta_index) {
        float term1 = first_terms(theta_index);
        float term2 = trace_estimates(theta_index);
        gradient(theta_index) = 0.5f * (term1 - term2);
    }
    if (DEBUG) std::cout << "Gradient: " << gradient.transpose() << std::endl;
}

// Compute second derivatives of the log-likelihood function wrt variance components (AI matrix)
void computeDerivative_AImat(const CohortDataStruct& CohortData, const VarCompDataStruct& VarCompData, const SolvingDataStruct& SolvingData, 
                             const MatrixXf& f_theta, MatrixXf& AI_matrix) {
    
    int num_varComps = VarCompData.num_varComps;
    int N = CohortData.N;
    AI_matrix.resize(num_varComps, num_varComps);

    // get V^{-1} f_theta
    MatrixXf Vinv_f_theta(N, num_varComps);
    if (DEBUG) std::cout << "Computing Vinv_f_theta (for Pf_theta -> AI matrix)... ";
    runPCG_multi_VC(CohortData, SolvingData, VarCompData, f_theta, Vinv_f_theta, 1e-5, 1000, false, true);
    
    // compute P f(theta)_j
    // MatrixXf Pf_theta = Vinv_f_theta; 
    // Pf_theta -= SolvingData.Vinv_X * SolvingData.Xt_Vinv_X_inv * CohortData.X.transpose() * Vinv_f_theta;
    MatrixXf temp1, temp2, temp3;
    MatrixXf Pf_theta = Vinv_f_theta;
    temp1.noalias() = CohortData.X.transpose() * Vinv_f_theta;
    temp2.noalias() = SolvingData.Xt_Vinv_X_inv * temp1;
    temp3.noalias() = SolvingData.Vinv_X * temp2;
    Pf_theta.noalias() -= temp3;

    // compute AI matrix
    AI_matrix.noalias() = 0.5f * f_theta.transpose() * Pf_theta;

    if (DEBUG) std::cout << "AI matrix: " << std::endl << AI_matrix << std::endl;
}


// Objective function to minimize (negative of the quadratic model)
double objective_function(const std::vector<double>& p, std::vector<double>& grad, void* data) {
    TrustRegionData* tr_data = static_cast<TrustRegionData*>(data);
    VectorXf gradient = tr_data->gradient;
    MatrixXf AI = tr_data->AI;

    int n = p.size();
    VectorXf p_vec(n);
    for (int i = 0; i < n; ++i) {
        p_vec(i) = p[i];
    }

    double obj = - (gradient.dot(p_vec) - 0.5 * p_vec.transpose() * AI * p_vec);

    if (!grad.empty()) {
        VectorXf grad_vec = - (gradient - AI * p_vec);
        for (int i = 0; i < n; ++i) {
            grad[i] = grad_vec(i);
        }
    }

    return obj;
}

// Trust region constraint: ||diag(AI) * p|| < Delta
double trust_region_constraint(const std::vector<double>& p, std::vector<double>& grad, void* data) {
    TrustRegionData* tr_data = static_cast<TrustRegionData*>(data);
    MatrixXf AI = tr_data->AI;
    float Delta = tr_data->Delta;

    int n = p.size();
    VectorXf diag_AI = AI.diagonal();
    VectorXf p_vec(n);
    for (int i = 0; i < n; ++i) {
        p_vec(i) = p[i];
    }
    double norm_AIp = (diag_AI.array() * p_vec.array()).sum();
    double constraint_value = norm_AIp - Delta;

    if (!grad.empty()) {
        VectorXf grad_vec = 2.0 * (diag_AI.array().square() * p_vec.array());
        for (int i = 0; i < n; ++i) {
            grad[i] = grad_vec(i);
        }
    }
    return constraint_value;
}

// Positivity constraint for each variance component: theta > epsilon
struct PositivityConstraintData {
    int idx;
    const VectorXf* theta_prev;
};
double positivity_constraint_vfunc(const std::vector<double>& p, std::vector<double>& grad, void* data) {
    const double scale_factor = 1e3; // scaling factor (to ensure constraint is respected under NLopt tol)
    PositivityConstraintData* constraint_data = static_cast<PositivityConstraintData*>(data);
    int idx = constraint_data->idx;
    const VectorXf& theta_prev = *(constraint_data->theta_prev);
    VectorXf p_vec(theta_prev.size());
    for (int i = 0; i < theta_prev.size(); ++i) {
        p_vec(i) = static_cast<float>(p[i]);
    }
    VectorXf theta_new = theta_prev + p_vec;
    double epsilon = 1e-5;
    double constraint_value = epsilon - theta_new(idx);    // Constraint: theta > eps --> eps - theta  < 0
    // std::cout << "Positivity constraint for theta_" << idx << ": " << constraint_value << std::endl;
    
    if (!grad.empty()) {
        grad.assign(grad.size(), 0.0);
        grad[idx] = -1.0 * scale_factor;
    }
    return constraint_value * scale_factor;
}


// Run one iteration of AI-REML with trust region optimization
void oneIter_AIREML(VectorXf& theta_prev, VarCompDataStruct& VarCompData, 
                    VectorXf& gradient_prev, MatrixXf& AI_matrix_prev,
                    CohortDataStruct& CohortData, SolvingDataStruct& SolvingData, 
                    float& Delta, bool& dangerous_step, int& num_rejections, 
                    bool& grad_check, const float grad_norm_threshold) {
    
    unsigned n = theta_prev.size();
    
    TrustRegionData tr_data;
    tr_data.gradient = gradient_prev;
    tr_data.AI = AI_matrix_prev;
    tr_data.Delta = Delta;
    tr_data.theta_prev = theta_prev;

    // Initial guess for p: zero step
    std::vector<double> p(n, 0.0);

    // Set up NLopt optimizer
    nlopt::opt opt(nlopt::LD_MMA, n);
    std::vector<double> lb(n, -10);
    std::vector<double> ub(n, 10);
    opt.set_maxeval(1e6);
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    opt.set_min_objective(objective_function, &tr_data);                          // Minimize the objective function
    
    // Apply constraints
    opt.add_inequality_constraint(trust_region_constraint, &tr_data, 1e-8);       // Trust region constraint

    std::vector<PositivityConstraintData> positivity_constraint_data(n);
    for (unsigned i = 0; i < n; ++i) {
        positivity_constraint_data[i].idx = i;
        positivity_constraint_data[i].theta_prev = &theta_prev;
        opt.add_inequality_constraint(positivity_constraint_vfunc, &positivity_constraint_data[i], 1e-10);
    }

    opt.set_xtol_rel(1e-10);
    opt.set_xtol_abs(1e-10);

    // Optimize
    double minf;
    nlopt::result result;
    try {
        result = opt.optimize(p, minf);
        std::cout << "NLopt result: " << result << ", minf = " << minf << std::endl;
    } catch (std::exception& e) {
        std::cout << "NLopt failed: " << e.what() << std::endl;
        Delta *= 0.75f; // Reduce trust region radius
        dangerous_step = true;
        num_rejections += 1;
        return;
    }
    if (result == 5) {
        std::cout << "NLopt optimization reach maxeval. Reducing trust region radius." << std::endl;
        Delta *= 0.75f; 
        dangerous_step = true;
        num_rejections += 1;
        return;
    }

    // Record predicted change in log-likelihood
    VectorXf p_vec(n);
    for (unsigned i = 0; i < n; ++i) {
        p_vec(i) = static_cast<float>(p[i]);
    }
    double delta_ll_pred = gradient_prev.dot(p_vec) - 0.5 * p_vec.transpose() * AI_matrix_prev * p_vec;
    std::cout << "Predicted change in log-likelihood: " << delta_ll_pred << std::endl;
    std::cout << "Predicted step: " << p_vec.transpose() << std::endl;
    
    // Initialize variables for potential step size reduction
    float scale = 1.0f;
    bool step_accepted = false;
    int max_reductions = 10;
    int reduction_count = 0;

    while (!step_accepted && reduction_count <= max_reductions) {
        VectorXf scaled_p_vec = scale * p_vec;
        VectorXf theta_new = theta_prev + scaled_p_vec;
        std::cout << "Trying step with scale " << scale << std::endl;
        std::cout << "theta_new: " << theta_new.transpose() << std::endl;

        // Check constraints on theta_new
        bool constraints_violated = false;
        for (unsigned i = 0; i < n; ++i) {
            if (theta_new(i) < 1e-6) {
                std::cout << "Warning: Negative variance component detected. Clipping." << std::endl;
                theta_new(i) = 1e-6f;
                constraints_violated = true;
            }
            if (theta_new(i) > 10.0f) {
                std::cout << "Warning: Large variance component detected. Clipping." << std::endl;
                theta_new(i) = 10.0f;
                constraints_violated = true;
            }
        }
        if (constraints_violated) {
            std::cout << "Constraints violated at scale " << scale << ". Reducing step size." << std::endl;
            scale *= 0.5f;
            reduction_count += 1;
            continue;
        }

        // Compute gradient at theta_new
        std::cout << "Computing gradient at theta_new..." << std::endl;
        VarCompDataStruct VarCompData_new;
        VarCompData_new.updateFromVector(theta_new);
        
        // Recompute model and gradient at new theta
        fitMME(CohortData, SolvingData, VarCompData_new);

        VectorXf gradient_new;
        MatrixXf f_theta_new;
        computeDerivative_first(CohortData, VarCompData_new, SolvingData, f_theta_new, gradient_new);

        // Compute approximate actual change in log-likelihood
        double delta_ll_approx = scaled_p_vec.dot(0.5 * (gradient_prev + gradient_new));
        double delta_ll_pred_scaled = gradient_prev.dot(scaled_p_vec) - 0.5 * scaled_p_vec.transpose() * AI_matrix_prev * scaled_p_vec;
        std::cout << "Predicted change in log-likelihood (scaled): " << delta_ll_pred_scaled << std::endl;
        std::cout << "(Approximate) Actual change in log-likelihood: " << delta_ll_approx << std::endl;

        // Check convergence
        const float tolLL = 1e-2;
        if (scale == 1 && delta_ll_pred_scaled < tolLL) {
            std::cout << "Converged: predicted log-likelihood improvement below threshold." << std::endl;
            dangerous_step = false;
            return;
        }

        // Check for dangerous model deviation
        double rho;
        if (grad_check && gradient_prev.norm() > grad_norm_threshold && gradient_new.norm() > 2 * gradient_prev.norm()) {
            std::cout << "Dangerous model deviation detected at scale " << scale << ". Reducing step size." << std::endl;
            scale *= 0.5f;
            reduction_count += 1;
            continue;
        } else {
            rho = delta_ll_approx / delta_ll_pred_scaled;
            std::cout << "Rho: " << rho << std::endl;
        }

        // Adjust trust region radius
        const float eta1 = 0.25f;
        const float eta2 = 0.99f;
        const float alpha1 = 0.25f;
        const float alpha2 = 3.5f;

        float norm_p_diag = 0.0f;
        for (unsigned i = 0; i < n; ++i) {
            float dpi = AI_matrix_prev(i, i) * scaled_p_vec(i);
            norm_p_diag += std::abs(dpi);
        }

        if (rho < eta1) {
            Delta = alpha1 * norm_p_diag;
        } else if (rho > eta2) {
            Delta = std::max(Delta, alpha2 * norm_p_diag);
        }
        
        // Decide whether to accept the step
        if (rho > eta1) {                  // Accept the step
            theta_prev = theta_new;
            VarCompData.updateFromVector(theta_new);
            gradient_prev = gradient_new;
            
            MatrixXf AI_matrix_new;
            computeDerivative_AImat(CohortData, VarCompData, SolvingData, f_theta_new, AI_matrix_new);
            AI_matrix_prev = AI_matrix_new;
            num_rejections = 0;
            step_accepted = true;
        } else {                           // Reject the step and reduce step size
            std::cout << "Step rejected at scale " << scale << ", rho = " << rho << ". Reducing step size." << std::endl;
            scale *= 0.5f;
            reduction_count += 1;
        }
    }
    if (!step_accepted) {
        std::cout << "Failed to accept a step after " << reduction_count << " reductions." << std::endl;
        Delta *= 0.75f; 
        dangerous_step = true;
        num_rejections += 1;
        return;
    }
}


int main(int argc, char* argv[]) {
    
    VarCompDataStruct VarCompData;
    CohortDataStruct CohortData;
    SolvingDataStruct SolvingData;

    // ============== 1. Process input files ===============
    // Collect cohort prameters
    std::string input_filename, grm_prefix, output_prefix; 
    parseCommandLineArgs(argc, argv, input_filename, grm_prefix, output_prefix, VarCompData);
    
    // redirect std::cout to log file
    std::ofstream log_file(output_prefix + ".log");
    std::streambuf* cout_buf = std::cout.rdbuf();
    TeeStreambuf tee_streambuf(cout_buf, log_file.rdbuf());
    std::cout.rdbuf(&tee_streambuf);

    std::cout << "Command: " << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl << std::endl;
    
    int nthreads = omp_get_max_threads();
    std::cout << "[main] OpenMP will utilize up to " << nthreads << " threads." << std::endl;

    checkInputs(CohortData, input_filename);
    
    // Prepare input matrices and read GRM
    size_t num_elems = static_cast<size_t>(CohortData.N) * (static_cast<size_t>(CohortData.N) + 1) / 2;

    CohortData.indiv.resize(CohortData.N);
    CohortData.Y.resize(CohortData.N);
    CohortData.A_vecLT.resize(num_elems);
    CohortData.X.resize(CohortData.N, CohortData.p);
    
    readInputData(CohortData, input_filename);
    // std::cout << "Matrices prepared successfully." << std::endl;
    readGRM(grm_prefix, CohortData);
    std::cout << "Input files read and processed successfully." << std::endl;

    
    // =============== 2. RUN AI-REML (Fitting + VC Estimation) ===============
    SolvingData.beta_hat.resize(CohortData.p);
    SolvingData.u_hat.resize(CohortData.N);
    
    VectorXf gradient;
    MatrixXf f_theta;
    MatrixXf AI_matrix;

    std::cout << "Running AI-REML..." << std::endl;
    std::cout << "Initial variance components: " << std::endl;
    std::cout << "  sig2e = " << VarCompData.sig2e << ", " << std::endl;
    std::cout << "  sig2g = " << VarCompData.sig2g << std::endl;
    
    VectorXf theta_prev(VarCompData.num_varComps);
    theta_prev << VarCompData.sig2e, VarCompData.sig2g;
    
    float Delta = 1e20;
    bool dangerous_step = false;
    int num_rejections = 0;
    bool grad_check = true;
    float grad_norm_threshold = 0;

    for (int i = 0; i < MAX_ITER; ++i) {
        std::string tline = (i < 9) ? "=================================" : "==================================";
        std::cout << std::endl << tline << std::endl;
        std::cout << "---------- Iteration " << i + 1 << " ----------" << std::endl;
        std::cout << tline << std::endl;
        auto iter_start_time = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            fitMME(CohortData, SolvingData, VarCompData);
            computeDerivative_first(CohortData, VarCompData, SolvingData, f_theta, gradient);
            computeDerivative_AImat(CohortData, VarCompData, SolvingData, f_theta, AI_matrix);
        }
        if (i == 1) {
            grad_norm_threshold = 0.1 * gradient.norm();
            std::cout << "Gradient norm threshold: " << grad_norm_threshold << std::endl;
        }
        if (num_rejections > 5) {
            std::cout << "Too many rejections at current point. Re-initalizing optimization." << std::endl;
            // sample new starting point uniformly from [0.1, 0.9]
            for (int j = 0; j < theta_prev.size(); ++j) {
                theta_prev(j) = 0.1f + 0.8f * static_cast<float>(rand()) / RAND_MAX;
            }
            VarCompData.updateFromVector(theta_prev);
            std::cout << "Re-initialized variance components:  sig2e = " << 
                VarCompData.sig2e << ", sig2g = " << 
                VarCompData.sig2g << std::endl;
            fitMME(CohortData, SolvingData, VarCompData);
            computeDerivative_first(CohortData, VarCompData, SolvingData, f_theta, gradient);
            computeDerivative_AImat(CohortData, VarCompData, SolvingData, f_theta, AI_matrix);
            num_rejections = 0;
        }
        
        VectorXf theta_old = theta_prev;
        oneIter_AIREML(theta_prev, VarCompData, gradient, AI_matrix, CohortData, SolvingData, Delta, dangerous_step, num_rejections, grad_check, grad_norm_threshold);
        std::cout << "Estimated variance components:  sig2e = " << 
            VarCompData.sig2e << ", sig2g = " << 
            VarCompData.sig2g << std::endl;
        
        auto iter_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> iter_duration = iter_end_time - iter_start_time;
        std::cout << "Iteration " << i + 1 << " took " << iter_duration.count() << " seconds." << std::endl;

        // Check for convergence
        float delta = (theta_prev - theta_old).norm() / theta_old.norm();
        if (delta < TOL && !dangerous_step) {
            std::cout << "Convergence reached after " << i + 1 << " iterations." << std::endl;
            break;
        }
        if (dangerous_step) {
            dangerous_step = false;
        }
        std::cout << "Convergence check: " << delta << std::endl;
        std::cout << "Trust region radius: " << Delta << std::endl;
        
        if (Delta < 1e-4) { // Termination condition for tiny trust region
            std::cout << "Trust region radius too small. Terminating optimization." << std::endl;
            break;
        }
    }

    // print final variance components and SEs (diagonals of -AI matrix)
    MatrixXf AI_inv = AI_matrix.inverse();
    std::cout << "Final variance components ± SEs:" << std::endl <<
        "   sig2e = " << VarCompData.sig2e << " ± " << sqrt(AI_inv(0, 0)) << std::endl <<
        "   sig2g = " << VarCompData.sig2g << " ± " << sqrt(AI_inv(1, 1)) << std::endl;
        
    std::cout.rdbuf(cout_buf); // Restore std::cout's original buffer
    log_file.close(); 
    
    return 0;
}

// compile: 
// g++ -O3 -g -fopenmp -lnlopt -I ~/TOOLS/nlopt/include/ -L ~/TOOLS/nlopt/lib64/ -I ~/TOOLS/ -o ~/tmp/varCompEst_reducedModel_noTime /n/home03/ahmadazim/TOOLS/GLOBE/src/varCompEst_reducedModel_noTime.cpp
// export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/TOOLS/nlopt/lib64/
// ~/tmp/varCompEst_reducedModel_noTime ~/tmp/input_5k.txt ~/WORKING/pilotMME/GRMs/grm_all_hm3_100k_val10k ~/tmp/output --sig2e 0.5 --sig2g 0.1
