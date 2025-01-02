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
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <bitset>
#include <sstream>
#include <cmath>
#include <algorithm>

// Type definitions
using namespace std;
using namespace Eigen;
using MatrixXf = Matrix<float, Dynamic, Dynamic>;
using VectorXf = Matrix<float, Dynamic, 1>;

// ========================================
//        STRUCTS AND GLOBAL OPTIONS
// ========================================

struct GlobalOptions {
    bool DEBUG = false;
    float MIN_MAF = 0.01f;
    float MAX_MISSING_RATE = 0.2f;
    bool PRECOMPUTE_GRM = false; 
};

struct CohortData {
    VectorXf indiv, Y;       // Size: Nrec
    MatrixXf X;              // Dimensions: Nrec x (p+1) (including TIME as last column)
    int p = 0, N = 0, Nrec = 0;
    std::string grm_prefix;
};

// Holds variance components, PCG settings, block preconditioner
struct SolvingData {
    float sig2e  = 0.1f; 
    float sig2g  = 0.1f; 
    float sig2b0 = 0.1f; 
    float sigb01 = 0.1f; 
    float sig2b1 = 0.1f;

    float PCG_TOL = 1e-5f;
    int PCG_MAX_ITER = 1000;

    // block diagonal preconditioner
    std::vector<int> block_starts;
    std::vector<std::vector<float>> Q_inv_blocks;
};

struct GenotypeData {
    size_t num_snps = 0;            // # filtered SNPs
    std::vector<float> allele_freqs;

    // Chunked storage of genotype bytes:
    //   Each SNP in a chunk occupies (N + 3)/4 bytes
    int numMarkersPerChunk = 0;
    int numChunks = 0;
    int remainderSNPs = 0;

    std::vector< std::vector<unsigned char>* > genoVecofPointers;

    VectorXf GRM_lt;  
};

inline size_t idxLowerTri(size_t i, size_t j) {
    if (i < j) std::swap(i, j);
    return (i*(i+1))/2 + j;
}

// Inline decode function (reverse 2 bits as defined in PLINK doc)
inline int decode2Bits(unsigned char code) {
    //  0b00 => homRef => 0
    //  0b10 => het => 1
    //  0b11 => homAlt => 2
    //  0b01 => missing => -1
    switch(code) {
        case 0b00: return 2; // homRef
        case 0b10: return 1; // het
        case 0b11: return 0; // homAlt
        case 0b01: default:  return -1; // missing
    }
}

struct ByteDecodeStats {
    unsigned char cnt00;
    unsigned char cnt10;
    unsigned char cnt01;
    unsigned char cnt11;
};

static ByteDecodeStats decodeLUT[256];
static bool LUT_initialized = false;

void initDecodeLUT() {
    if (LUT_initialized) return;

    for (int b = 0; b < 256; b++) {
        unsigned char cnt00 = 0, cnt10 = 0, cnt01 = 0, cnt11 = 0;
        for (int pair_idx = 0; pair_idx < 4; pair_idx++) {
            unsigned char code = (b >> (2*pair_idx)) & 0x03;
            switch(code) {
                case 0b00: cnt00++; break; // homRef => alt_count +=2
                case 0b10: cnt10++; break; // het => alt_count +=1
                case 0b01: cnt01++; break; // missing
                case 0b11: cnt11++; break; // homAlt => alt_count +=0
            }
        }
        decodeLUT[b] = {cnt00, cnt10, cnt01, cnt11};
    }
    LUT_initialized = true;
}

// ========================================
//           FUNCTION DECLARATIONS
// ========================================

void parseCommandLineArgs(int argc, char* argv[],
                          GlobalOptions& gOpt,
                          CohortData& cData,
                          SolvingData& solveData,
                          std::string& input_filename,
                          std::string& output_prefix,
                          std::string& geno_prefix);

void checkInputs(const std::string& filename, 
                 CohortData& cData,
                 bool debug_mode);

void readInputData(CohortData& cData, 
                   const std::string& filename, 
                   bool debug_mode);

void prepGenotypeInput(std::string geno_prefix,
                       CohortData& cData,
                       GenotypeData& gData, 
                       float maf_threshold, 
                       float missing_threshold, 
                       bool debug_mode);

// "placeholder"
void precomputeStandardizedGenotypes(const std::string& bedfile, 
                                     CohortData& cData,
                                     GenotypeData& gData, 
                                     bool debug_mode, 
                                     bool precomputeGRM);

void prepareAllInputs(CohortData& cData, 
                      GenotypeData& gData,
                      const std::string& input_filename,
                      std::string geno_prefix,
                      const GlobalOptions& gOpt);

// void computeFullGRM(CohortData& cData,
//                     GenotypeData& gData,
//                     bool debug_mode);
void readFullGRM(CohortData& cData,
                 GenotypeData& gData,
                 bool debug_mode);

inline float getElement_GRM(size_t i, size_t j, 
                            const CohortData& cData,
                            const GenotypeData& gData,
                            bool precomputeGRM);

int getGenotype(const size_t snp_idx, 
                const int indiv_idx,
                const CohortData& cData,
                const GenotypeData& gData);

float getStdGenotype(size_t snp_idx, 
                     int indiv_idx,
                     const CohortData& cData,
                     const GenotypeData& gData);

// float computeV_ij(int i, int j, 
//                   const CohortData& cData, 
//                   const GenotypeData& gData,
//                   const SolvingData& solveData,
//                   bool precomputeGRM);

void computePreconditioner_Block(const CohortData& cData,
                                 const GenotypeData& gData,
                                 SolvingData& solveData, 
                                 const GlobalOptions& gOpt);

void applyPreconditioner(const SolvingData& solveData,
                         const VectorXf& r,
                         VectorXf& z);

void computeVx_multi(const MatrixXf& input_matrix, 
                     MatrixXf& result_matrix,
                     const CohortData& cData, 
                     const GenotypeData& gData,
                     const SolvingData& solveData, 
                     const GlobalOptions& gOpt,
                     bool multGRM_uhat);

void runPCG_multi(const MatrixXf& b, 
                  MatrixXf& x,
                  const CohortData& cData,
                  const GenotypeData& gData,
                  SolvingData& solveData,
                  const GlobalOptions& gOpt);

void compute_uhat(const VectorXf& Vinv_r, 
                  VectorXf& u_hat, 
                  const CohortData& cData, 
                  const GenotypeData& gData,
                  const SolvingData& solveData, 
                  const GlobalOptions& gOpt);

void compute_bhat(const VectorXf& Vinv_r, 
                  VectorXf& b_hat, 
                  const CohortData& cData,
                  const SolvingData& solveData, 
                  bool debug_mode);


// ========================================
//               MAIN FUNCTION
// ========================================

int main(int argc, char* argv[]) 
{
    // 1. Parse arguments
    GlobalOptions gOpt;
    SolvingData solveData;
    CohortData cData;
    GenotypeData gData;
    std::string input_filename, geno_prefix, output_prefix;
    parseCommandLineArgs(argc, argv, gOpt, cData, solveData, input_filename, output_prefix, geno_prefix);

    int nthreads = omp_get_max_threads();
    std::cout << "[main] OpenMP will utilize up to " << nthreads << " threads." << std::endl;

    // 2. Collect params from input files
    checkInputs(input_filename, cData, gOpt.DEBUG);

    // 3. Prepare data structs for phenotype/covariate
    cData.indiv.resize(cData.Nrec);
    cData.Y.resize(cData.Nrec);
    cData.X.resize(cData.Nrec, cData.p + 1);

    // 4. Organize inputs
    prepareAllInputs(cData, gData, input_filename, geno_prefix, gOpt);

    // 5. Build the block-diagonal preconditioner
    computePreconditioner_Block(cData, gData, solveData, gOpt);

    // 6. Run PCG solver
    MatrixXf XY(cData.Nrec, cData.p + 2);
    XY << cData.X, cData.Y;
    MatrixXf Vinv_XY(cData.Nrec, cData.p + 2);

    runPCG_multi(XY, Vinv_XY, cData, gData, solveData, gOpt);

    MatrixXf Vinv_X = Vinv_XY.leftCols(cData.p + 1);
    VectorXf Vinv_Y = Vinv_XY.rightCols(1);
    
    if (gOpt.DEBUG) {
        std::cout << "First 5 rows of Vinv_X:" << std::endl;
        std::cout << Vinv_X.topRows(5) << std::endl;
        std::cout << "First 5 elements of Vinv_Y:" << std::endl;
        std::cout << Vinv_Y.head(5) << std::endl;
    }

    MatrixXf Xt_Vinv_X = cData.X.transpose() * Vinv_X;
    MatrixXf Xt_Vinv_X_inv = Xt_Vinv_X.inverse();
    VectorXf beta_hat = Xt_Vinv_X_inv * (cData.X.transpose() * Vinv_Y);

    VectorXf r = cData.Y - cData.X * beta_hat;
    VectorXf Vinv_r = Vinv_Y - Vinv_X * beta_hat;
    
    if (gOpt.DEBUG) {
        std::cout << "First 5 elements of Vinv_r:" << std::endl;
        std::cout << Vinv_r.head(5) << std::endl;
    }

    // Genetic random effect
    VectorXf u_hat(cData.N);
    compute_uhat(Vinv_r, u_hat, cData, gData, solveData, gOpt);

    // Longitudinal random effects
    VectorXf b_hat(2 * cData.N);
    compute_bhat(Vinv_r, b_hat, cData, solveData, gOpt.DEBUG);

    // Save
    saveVector(beta_hat, output_prefix + "_beta.bin");
    saveVector(u_hat,   output_prefix + "_uhat.bin");
    saveVector(b_hat,   output_prefix + "_bhat.bin");

    std::cout << "[main] Done." << std::endl;
    return 0;
}


// ========================================
//          FUNCTION DEFINITIONS
// ========================================

void parseCommandLineArgs(
    int argc, char* argv[],
    GlobalOptions& gOpt,
    CohortData& cData,
    SolvingData& solveData,
    std::string& input_filename,
    std::string& output_prefix,
    std::string& geno_prefix)
{
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << argv[0] << " --input <file> --output <prefix> "
                  << " --sig2e <val> --sig2g <val> --sig2b0 <val> --sigb01 <val> --sig2b1 <val>\n"
                  << "  EITHER:  --geno <plinkPrefix>\n"
                  << "      OR:  --grm <grmPrefix>\n"
                  << " Optional:\n"
                  << "   [--debug] [--minMAF val] [--maxMissingRate val]\n\n"
                  << "Example:\n"
                  << "  " << argv[0] << " \\\n"
                  << "     --input data.txt \\\n"
                  << "     --output results \\\n"
                  << "     --geno /path/to/myPlink \\\n"
                  << "     --sig2e 0.6 --sig2g 0.4 --sig2b0 0.4 --sigb01 0.2 --sig2b1 0.2 \\\n"
                  << "     --debug\n"
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    bool haveSig2e  = false;
    bool haveSig2g  = false;
    bool haveSig2b0 = false;
    bool haveSigb01 = false;
    bool haveSig2b1 = false;
    bool haveInput  = false;
    bool haveOutput = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto nextValue = [&](bool required=true) -> std::string {
            if (i + 1 >= argc) {
                if (required) {
                    std::cerr << "[Error] Flag " << arg << " expects an argument.\n";
                    exit(EXIT_FAILURE);
                }
                return std::string(); // empty if not required
            }
            return std::string(argv[++i]);
        };

        if (arg == "--debug") {
            gOpt.DEBUG = true;
        }
        else if (arg == "--minMAF") {
            std::string val = nextValue();
            gOpt.MIN_MAF = std::stof(val);
        }
        else if (arg == "--maxMissingRate") {
            std::string val = nextValue();
            gOpt.MAX_MISSING_RATE = std::stof(val);
        }
        else if (arg == "--geno") {
            // PLINK prefix
            geno_prefix = nextValue();
        }
        else if (arg == "--grm") {
            // Precomputed GRM prefix
            gOpt.PRECOMPUTE_GRM = true;
            cData.grm_prefix    = nextValue();
        }
        else if (arg == "--input") {
            input_filename = nextValue();
            haveInput = true;
        }
        else if (arg == "--output") {
            output_prefix = nextValue();
            haveOutput = true;
        }
        else if (arg == "--sig2e") {
            solveData.sig2e = std::stof(nextValue());
            haveSig2e = true;
        }
        else if (arg == "--sig2g") {
            solveData.sig2g = std::stof(nextValue());
            haveSig2g = true;
        }
        else if (arg == "--sig2b0") {
            solveData.sig2b0 = std::stof(nextValue());
            haveSig2b0 = true;
        }
        else if (arg == "--sigb01") {
            solveData.sigb01 = std::stof(nextValue());
            haveSigb01 = true;
        }
        else if (arg == "--sig2b1") {
            solveData.sig2b1 = std::stof(nextValue());
            haveSig2b1 = true;
        }
        else {
            // Unknown flag
            std::cerr << "[Error] Unknown flag: " << arg << "\n";
            exit(EXIT_FAILURE);
        }
    }

    // Validate that we got all the required parameters
    if (!haveInput || !haveOutput 
        || !haveSig2e || !haveSig2g || !haveSig2b0 
        || !haveSigb01 || !haveSig2b1) 
    {
        std::cerr << "[Error] Missing required flags.\n";
        exit(EXIT_FAILURE);
    }

    // Check MAF/missing thresholds
    if (gOpt.MIN_MAF < 0 || gOpt.MIN_MAF > 0.5f
        || gOpt.MAX_MISSING_RATE < 0 || gOpt.MAX_MISSING_RATE > 1.0f)
    {
        std::cerr << "[Error] Invalid MAF or missing rate thresholds.\n";
        exit(EXIT_FAILURE);
    }

    // Check that EXACTLY one of (--geno) or (--grm) is used
    bool haveGeno = !geno_prefix.empty();
    bool haveGRM  = !cData.grm_prefix.empty();

    if ((haveGeno && haveGRM) || (!haveGeno && !haveGRM)) {
        std::cerr << "[Error] You must specify exactly one of:\n"
                  << "  --geno <plinkPrefix>\n"
                  << "OR\n"
                  << "  --grm  <grmPrefix>\n"
                  << "but not both.\n";
        exit(EXIT_FAILURE);
    }

    // Debug info
    if (gOpt.DEBUG) {
        std::cout << "[parseCommandLineArgs]"
                  << "\n  input=" << input_filename
                  << "\n  output=" << output_prefix
                  << "\n  geno_prefix=" << geno_prefix
                  << "\n  cData.grm_prefix=" << cData.grm_prefix
                  << "\n  sig2e=" << solveData.sig2e
                  << ", sig2g=" << solveData.sig2g
                  << ", sig2b0=" << solveData.sig2b0
                  << ", sigb01=" << solveData.sigb01
                  << ", sig2b1=" << solveData.sig2b1
                  << "\n  minMAF=" << gOpt.MIN_MAF 
                  << ", maxMissingRate=" << gOpt.MAX_MISSING_RATE
                  << "\n  PRECOMPUTE_GRM=" << (gOpt.PRECOMPUTE_GRM ? "TRUE" : "FALSE")
                  << std::endl;
    }
}

void checkInputs(const std::string& filename, 
                 CohortData& cData,
                 bool debug_mode)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[checkInputs] Error opening input file: " 
                  << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    std::getline(file, line); // header line
    std::istringstream header(line);
    std::string column;

    int indivIndex = -1, phenoIndex = -1, timesIndex = -1;
    int index = 0;
    while (header >> column) {
        if (column == "INDIV") {
            indivIndex = index;
        } else if (column == "PHENO") {
            phenoIndex = index;
        } else if (column == "TIMES") {
            timesIndex = index;
        }
        ++index;
    }
    if (indivIndex == -1 || phenoIndex == -1 || timesIndex == -1) {
        std::cerr << "[checkInputs] Error: Missing required columns in header." << std::endl;
        exit(EXIT_FAILURE);
    }
    cData.p = index - 3;

    int Nrec = 0;
    std::unordered_set<int> unique_indivs;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float indiv_id;
        for (int i = 0; i < index; ++i) {
            if (i == indivIndex) {
                iss >> indiv_id;
                unique_indivs.insert((int) indiv_id);
            } else {
                float tmp;
                iss >> tmp; 
            }
        }
        ++Nrec;
    }
    file.close();

    cData.N = unique_indivs.size();
    cData.Nrec = Nrec;

    if (debug_mode) {
        std::cout << "[checkInputs] Found:\n"
                  << "   N = " << cData.N << " unique individuals\n"
                  << "   p = " << cData.p << " fixed effects (not counting TIME)\n"
                  << "   Nrec = " << cData.Nrec << " records total." << std::endl;
    }
}

void readInputData(CohortData& cData, 
                   const std::string& filename, 
                   bool debug_mode)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[readInputData] Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string line;
    std::getline(file, line); // header
    std::istringstream header_line(line);
    std::string column;

    int indivIndex = -1, phenoIndex = -1, timesIndex = -1;
    int index = 0;
    while (header_line >> column) {
        if (column == "INDIV") {
            indivIndex = index;
        } else if (column == "PHENO") {
            phenoIndex = index;
        } else if (column == "TIMES") {
            timesIndex = index;
        }
        ++index;
    }

    int row = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float indiv_id = -1.f, pheno = 0.f, time = 0.f;
        for (int i = 0; i < index; ++i) {
            if (i == indivIndex) {
                iss >> indiv_id;
            } else if (i == phenoIndex) {
                iss >> pheno;
            } else if (i == timesIndex) {
                iss >> time;
            } else {
                cData.X(row, i - 3) = 0.0f;
                iss >> cData.X(row, i - 3);
            }
        }
        cData.indiv(row) = indiv_id - 1; // 0-index
        cData.Y(row)     = pheno;
        cData.X(row, cData.p) = time;
        ++row;
    }
    file.close();

    if (debug_mode) {
        std::cout << "[readInputData] First 5 rows of X:\n"
                  << cData.X.topRows(5) << std::endl
                  << "[readInputData] First 5 elements of Y:\n"
                  << cData.Y.head(5).transpose() << std::endl
                  << "[readInputData] First 5 elements of indiv:\n"
                  << cData.indiv.head(5).transpose() << std::endl;
    }
}

// Utility to handle partial last byte
void subtractPhantomGenotypes(unsigned char last_byte,
                              int phantom_pairs,
                              size_t& alt_count,
                              size_t& missing_count)
{
    for (int p = 0; p < phantom_pairs; p++) {
        int pair_idx = 3 - p;
        unsigned char code = (last_byte >> (2 * pair_idx)) & 0x03;
        switch(code) {
            case 0b00:
                alt_count -= 2;
                break;
            case 0b10:
                alt_count -= 1;
                break;
            case 0b01:
                missing_count -= 1;
                break;
            case 0b11:
            default:
                break;
        }
    }
}

void prepGenotypeInput(std::string geno_prefix,
                       CohortData& cData,
                       GenotypeData& gData, 
                       float maf_threshold, 
                       float missing_threshold, 
                       bool debug_mode)
{
    initDecodeLUT();

    std::string bedfile = geno_prefix + ".bed";
    std::string bimfile = geno_prefix + ".bim";
    std::string famfile = geno_prefix + ".fam";
    int N = cData.N;

    std::ifstream bim(bimfile);
    if (!bim.is_open()) {
        std::cerr << "[prepGenotypeInput] Error opening .bim file: " << bimfile << std::endl;
        exit(EXIT_FAILURE);
    }
    size_t total_snps = std::count(std::istreambuf_iterator<char>(bim),
                                   std::istreambuf_iterator<char>(),
                                   '\n');
    bim.close();

    if (debug_mode) {
        std::cout << "[prepGenotypeInput] total_snps from .bim = " 
                  << total_snps << std::endl;
    }

    std::ifstream bed(bedfile, std::ios::binary);
    if (!bed.is_open()) {
        std::cerr << "[prepGenotypeInput] Error opening .bed file: " 
                  << bedfile << std::endl;
        exit(EXIT_FAILURE);
    }
    unsigned char magic[3];
    bed.read(reinterpret_cast<char*>(magic), 3);
    if (magic[0] != 0x6C || magic[1] != 0x1B) {
        std::cerr << "[prepGenotypeInput] Invalid .bed format (magic mismatch)." 
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    if (magic[2] == 0x00) {
        std::cerr << "[prepGenotypeInput] BED is individual-major, code expects SNP-major => aborting." 
                  << std::endl;
        exit(EXIT_FAILURE);
    } else if (magic[2] != 0x01) {
        std::cerr << "[prepGenotypeInput] Third byte unknown => aborting." << std::endl;
        exit(EXIT_FAILURE);
    }

    // read genotype bytes and chunk
    size_t bytes_per_snp = (N + 3) / 4;
    std::vector<unsigned char> g_buffer(bytes_per_snp);

    double memoryChunkGB = 1.0; // TODO: user-defined
    size_t numMarkersPerChunk = static_cast<size_t>(
        (memoryChunkGB * 1e9) / bytes_per_snp
    );
    if (numMarkersPerChunk == 0) {
        numMarkersPerChunk = 50000;
    }
    gData.numMarkersPerChunk = (int)numMarkersPerChunk;

    int numFullChunks = (int)(total_snps / numMarkersPerChunk);
    int remainder = (int)(total_snps % numMarkersPerChunk);
    gData.numChunks = (remainder == 0) ? numFullChunks : numFullChunks + 1;
    gData.remainderSNPs = remainder;

    if (debug_mode) {
        std::cout << "[prepGenotypeInput] We will have " << gData.numChunks 
                  << " chunk(s) for " << total_snps << " total SNPs.\n"
                  << "  each chunk can store up to " << numMarkersPerChunk 
                  << " SNPs.\n";
    }
    gData.genoVecofPointers.resize(gData.numChunks, nullptr);
    for (int c = 0; c < gData.numChunks; c++) {
        gData.genoVecofPointers[c] = new std::vector<unsigned char>();
        size_t chunkSize = (c < numFullChunks) 
                         ? (numMarkersPerChunk * bytes_per_snp)
                         : (remainder * bytes_per_snp);
        gData.genoVecofPointers[c]->reserve(chunkSize);
    }

    std::vector<float> alleleFreqsTemp;
    alleleFreqsTemp.reserve(total_snps);

    int remainder_in_last_byte = N % 4;
    int phantom_count = (remainder_in_last_byte == 0) ? 0 : (4 - remainder_in_last_byte);

    size_t kept_snps = 0;
    for (size_t snp_i = 0; snp_i < total_snps; snp_i++) 
    {
        // read bytes for SNP
        bed.read(reinterpret_cast<char*>(g_buffer.data()), bytes_per_snp);

        size_t alt_count = 0;
        size_t missing_count = 0;

        // decode
        #pragma omp parallel for reduction(+:alt_count, missing_count)
        for (int byte_idx = 0; byte_idx < (int)bytes_per_snp; byte_idx++) {
            unsigned char b = g_buffer[byte_idx];
            ByteDecodeStats st = decodeLUT[b];
            alt_count += 2*st.cnt00 + st.cnt10;
            missing_count += st.cnt01;
        }

        // handle partial byte if needed
        if (phantom_count > 0) subtractPhantomGenotypes(g_buffer[bytes_per_snp-1], phantom_count, alt_count, missing_count);

        int n_obs = N - missing_count;
        float maf = (n_obs > 0) ? (float)alt_count / (2.0f * n_obs) : 0.0f;
        float missRate = (float)missing_count / (float)N;

        // filter on MAF and missing
        if (maf >= maf_threshold && maf <= (1.0f - maf_threshold) 
            && missRate <= missing_threshold) 
        {
            alleleFreqsTemp.push_back(maf);

            int chunk_idx = (int)(kept_snps / numMarkersPerChunk);
            gData.genoVecofPointers[chunk_idx]->insert(
                gData.genoVecofPointers[chunk_idx]->end(),
                g_buffer.begin(),
                g_buffer.end()
            );
            kept_snps++;
        }
        // Progress tracker
        if (snp_i % 100000 == 0 || snp_i == total_snps - 1) {
            std::cout << "[prepGenotypeInput] Processed " << snp_i + 1 
                      << " / " << total_snps << " SNPs." << std::endl;
        }
    }

    bed.close();

    gData.num_snps = kept_snps;
    gData.allele_freqs.resize(kept_snps);
    for (size_t i = 0; i < kept_snps; i++) {
        gData.allele_freqs[i] = alleleFreqsTemp[i];
    }

    if (debug_mode) {
        std::cout << "[prepGenotypeInput] Filtered SNPs = " 
                  << kept_snps << " / " << total_snps << std::endl;
    }
}

void precomputeStandardizedGenotypes(const std::string& bedfile, 
                                     CohortData& cData,
                                     GenotypeData& gData, 
                                     bool debug_mode, 
                                     bool GRM_precompute)
{
    int N = cData.N;
    if (debug_mode) {
        std::cout << "[precomputeStandardizedGenotypes] Not performing full matrix expansion."
            << (GRM_precompute ? " We'll precompute the GRM." : " We'll decode on-the-fly.") << std::endl;
    }
}

void readFullGRM(CohortData& cData,
                 GenotypeData& gData,
                 bool debug_mode)
{
    const std::string& grm_filename = cData.grm_prefix + ".grm.bin";
    size_t nelt = (size_t)cData.N * (cData.N + 1) / 2;
    gData.GRM_lt.resize(nelt);

     std::ifstream file(grm_filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[readFullGRM] Error opening GRM file: " << grm_filename << std::endl;
        exit(EXIT_FAILURE);
    }
    file.read(reinterpret_cast<char*>(gData.GRM_lt.data()), nelt * sizeof(float));
    file.close();

    if (!file) {
        std::cerr << "[readFullGRM] Error reading from file: " << grm_filename << std::endl;
        exit(EXIT_FAILURE);
    }

    if (debug_mode) {
        std::cout << "[readFullGRM] First 5 rows & columns of GRM:\n";
        for (int i = 0; i < std::min(5, cData.N); ++i) {
            for (int j = 0; j <= i; ++j) {
                size_t idx = (size_t)i*(i+1)/2 + j;
                std::cout << gData.GRM_lt[idx] << " ";
            }
            std::cout << std::endl;
        }
    }
}

// get genotype on the fly
int getGenotype(const size_t snp_idx, 
                const int indiv_idx,
                const CohortData& cData,
                const GenotypeData& gData)
{
    int chunk_idx = (int)(snp_idx / gData.numMarkersPerChunk);
    int snp_offset_in_chunk = (int)(snp_idx % gData.numMarkersPerChunk);

    size_t bytes_per_snp = (cData.N + 3) / 4;
    size_t base_offset = snp_offset_in_chunk * bytes_per_snp;

    auto* chunkPtr = gData.genoVecofPointers[chunk_idx];

    size_t byte_idx = indiv_idx / 4;
    size_t bit_offset = (indiv_idx % 4) * 2;
    unsigned char code = ((*chunkPtr)[base_offset + byte_idx] >> bit_offset) & 0x03;
    return decode2Bits(code);
}

float getStdGenotype(size_t snp_idx, 
                     int indiv_idx,
                     const CohortData& cData,
                     const GenotypeData& gData)
{
    int g = getGenotype(snp_idx, indiv_idx, cData, gData);
    if (g < 0) g = 0; // treat missing as 0
    float maf = gData.allele_freqs[snp_idx];
    float denom = 2.0f * maf * (1.0f - maf);
    float inv_std = (denom > 1e-12f) ? (1.0f / std::sqrt(denom)) : 1.0f;
    return (g - 2.0f * maf) * inv_std;
}

void prepareAllInputs(CohortData& cData, 
                      GenotypeData& gData,
                      const std::string& input_filename,
                      std::string geno_prefix,
                      const GlobalOptions& gOpt)
{
    readInputData(cData, input_filename, gOpt.DEBUG);

    if (!gOpt.PRECOMPUTE_GRM) {
        prepGenotypeInput(geno_prefix,
                          cData, 
                          gData, 
                          gOpt.MIN_MAF, 
                          gOpt.MAX_MISSING_RATE, 
                          gOpt.DEBUG);
        
        precomputeStandardizedGenotypes(geno_prefix + ".bed", cData, gData, gOpt.DEBUG, gOpt.PRECOMPUTE_GRM);
    } else {
        readFullGRM(cData, gData, gOpt.DEBUG);
    }

    if (gOpt.DEBUG && gData.num_snps > 0) {
        std::cout << "[prepareAllInputs] First 5 rows & columns of GRM:\n";
        for (size_t i = 0; i < std::min<size_t>(5, cData.N); i++) {
            for (size_t j = 0; j <= i; j++) {
                float val = getElement_GRM(i, j, cData, gData, gOpt.PRECOMPUTE_GRM);
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "[prepareAllInputs] Input prepared successfully." << std::endl;
}

inline size_t ltIndex(int i, int j) {
    if (j > i) std::swap(i, j);
    return (size_t)i*(i+1)/2 + j;
}

inline float getElement_GRM(size_t i, size_t j, 
                            const CohortData& cData,
                            const GenotypeData& gData,
                            bool precomputeGRM)
{
    if (!precomputeGRM) {
        float val = 0.0f;
        #pragma omp parallel for reduction(+:val)
        for (size_t snp = 0; snp < gData.num_snps; snp++) {
            float gi = getStdGenotype(snp, (int)i, cData, gData);
            float gj = getStdGenotype(snp, (int)j, cData, gData);
            val += gi * gj;
        }
        if (gData.num_snps > 0) val /= (float)gData.num_snps;
        return val;
    } 
    else {
        size_t idx = ltIndex((int)i, (int)j);
        return gData.GRM_lt[idx];
    }
}

// float computeV_ij(int i, int j, 
//                   const CohortData& cData, 
//                   const GenotypeData& gData,
//                   const SolvingData& solveData,
//                   bool precomputeGRM)
// {
//     int indiv_i = (int)cData.indiv(i);
//     int indiv_j = (int)cData.indiv(j);

//     float ti = cData.X(i, cData.p);
//     float tj = cData.X(j, cData.p);

//     float genetic_component = solveData.sig2g * getElement_GRM(indiv_i, indiv_j, cData, gData, precomputeGRM);
//     float temporal_component = (indiv_i == indiv_j) ? (solveData.sig2b0 + solveData.sigb01*(ti + tj) + solveData.sig2b1*(ti * tj)) : 0.0f;
//     float residual_component = ((i == j) ? solveData.sig2e : 0.0f);
    
//     return (genetic_component + temporal_component + residual_component);
// }

// Compute diagonal GRM entries
std::vector<float> computeGeneticDiagonal(const CohortData& cData,
                                          const GenotypeData& gData,
                                          const GlobalOptions& gOpt)
{
    const bool precomputeGRM = gOpt.PRECOMPUTE_GRM;
    const bool debug_mode = gOpt.DEBUG;
    const int N = cData.N;

    std::vector<float> gVar(N, 0.0f);
    if (gData.num_snps == 0) return gVar; 

    if (precomputeGRM) {
        for (int i = 0; i < N; i++) {
            size_t idx = (size_t)i * (i+1)/2 + i;
            gVar[i] = gData.GRM_lt[idx];
        }
        if (debug_mode) {
            std::cout << "[computeGeneticDiagonal] Using precomputed GRM for diagonal." << std::endl;
        }
        return gVar;
    }
    
    if (debug_mode) {
        std::cout << "[computeGeneticDiagonal] Computing diagonal on the fly..." << std::endl;
    }

    size_t num_snps = gData.num_snps;
    const float inv_num_snp = (num_snps > 0) ? (1.0f / (float)num_snps) : 0.0f;

    const size_t SNP_CHUNK_SIZE = 50000;  // TODO: user-defined
    size_t snp_start = 0;
    while (snp_start < num_snps)
    {
        size_t snp_end = std::min(snp_start + SNP_CHUNK_SIZE, num_snps);
        #pragma omp parallel
        {
            std::vector<float> local_buf(N, 0.0f);

            #pragma omp for schedule(dynamic)
            for (int s = (int)snp_start; s < (int)snp_end; ++s) {
                float maf   = gData.allele_freqs[s];
                float denom = 2.0f * maf * (1.0f - maf);
                float inv_std = (denom > 1e-12f) ? (1.0f / std::sqrt(denom)) : 1.0f;

                for (int i = 0; i < N; i++) {
                    int genotype = getGenotype(s, i, cData, gData);
                    if (genotype < 0) genotype = 0; // missing => treat as 0
                    float g_std = (genotype - 2.0f * maf) * inv_std;
                    local_buf[i] += (g_std * g_std); 
                }
            }
            #pragma omp critical
            {
                for (int i = 0; i < N; i++) {
                    gVar[i] += local_buf[i];
                }
            }
        }
        snp_start = snp_end;
    }
    for (int i = 0; i < N; i++) {
        gVar[i] *= inv_num_snp;
    }
    return gVar;
}

void computePreconditioner_Block(const CohortData& cData,
                                 const GenotypeData& gData,
                                 SolvingData& solveData, 
                                 const GlobalOptions& gOpt)
{
    const bool precomputeGRM = gOpt.PRECOMPUTE_GRM;
    const bool debug_mode = gOpt.DEBUG;

    std::vector<int>& block_starts = solveData.block_starts;
    std::vector<std::vector<float>>& Q_inv_blocks = solveData.Q_inv_blocks;
    int Nrec = cData.Nrec;
    const auto& indiv = cData.indiv;

    block_starts.clear();
    block_starts.push_back(0);
    for (int i = 1; i < Nrec; i++) {
        if (indiv[i] != indiv[i-1]) {
            block_starts.push_back(i);
        }
    }
    block_starts.push_back(Nrec);
    
    std::vector<float> gVar = computeGeneticDiagonal(cData, gData, gOpt);

    int num_blocks = (int) block_starts.size() - 1;
    Q_inv_blocks.resize(num_blocks);

    // for each block b => build and invert
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < num_blocks; b++) {
        int start = block_starts[b];
        int end   = block_starts[b+1];
        int block_size = end - start;

        int u = (int) indiv[start];
        float gVal = solveData.sig2g * gVar[u];
        MatrixXf blockM(block_size, block_size);
        for (int i = 0; i < block_size; i++) {
            int row_i = start + i;
            float ti = cData.X(row_i, cData.p);
            for (int j = 0; j <= i; j++) {
                int row_j = start + j;
                float tj = cData.X(row_j, cData.p);
                float timePart = (solveData.sig2b0 + solveData.sigb01 * (ti + tj) + solveData.sig2b1  * (ti * tj));
                float val = gVal + timePart;
                if (i == j) {
                    val += solveData.sig2e;
                }
                blockM(i, j) = val;
                if (i != j) {
                    blockM(j, i) = val;
                }
            }
        }
        MatrixXf block_inv = blockM.inverse();
        std::vector<float> block_inv_LT(block_size*(block_size+1)/2, 0.0f);
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j <= i; j++) {
                block_inv_LT[i*(i+1)/2 + j] = block_inv(i, j);
            }
        }
        Q_inv_blocks[b] = std::move(block_inv_LT);
    }
    if (debug_mode) {
        std::cout << "[computePreconditioner_Block] Built " << num_blocks 
                  << " blocks for approximate preconditioner." << std::endl;
    }
}

void applyPreconditioner(const SolvingData& solveData,
                         const VectorXf& r,
                         VectorXf& z)
{
    z.resize(r.size());
    const auto& block_starts = solveData.block_starts;
    const auto& Q_inv_blocks = solveData.Q_inv_blocks;

    #pragma omp parallel for
    for (int b = 0; b < (int)block_starts.size() - 1; ++b) {
        int start = block_starts[b];
        int end   = block_starts[b+1];
        int block_size = end - start;

        VectorXf r_block = r.segment(start, block_size);
        VectorXf z_block(block_size); 
        z_block.setZero();

        const std::vector<float>& blockLT = Q_inv_blocks[b];
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j <= i; ++j) {
                float val = blockLT[i*(i+1)/2 + j];
                z_block[i] += val * r_block[j];
                if (i != j) {
                    z_block[j] += val * r_block[i];
                }
            }
        }
        #pragma omp critical
        {
            z.segment(start, block_size) = z_block;
        }
    }
}

void computeVx_multi(const MatrixXf& input_matrix, 
                     MatrixXf& result_matrix,
                     const CohortData& cData, 
                     const GenotypeData& gData,
                     const SolvingData& solveData, 
                     const GlobalOptions& gOpt,
                     bool multGRM_uhat)
{

    const int Nrec    = cData.Nrec;
    const int Nindiv  = cData.N;
    const int num_rhs = input_matrix.cols();

    if (multGRM_uhat){
        result_matrix.setZero(Nindiv, num_rhs);
    } else {
        result_matrix.setZero(Nrec, num_rhs);
    }

    // ------------------------------------------------------------------
    // A) GENETIC PART => J * GRM * J^T * X
    // ------------------------------------------------------------------
    
    // Step 1: Z = J^T * X  (record -> individual space)
    MatrixXf Z = MatrixXf::Zero(Nindiv, num_rhs);
    for(int r = 0; r < Nrec; r++) {
        int i = (int)cData.indiv(r);
        for(int col = 0; col < num_rhs; col++) {
            Z(i, col) += input_matrix(r, col);
        }
    }

    // Step 2: GZ = GRM * Z
    MatrixXf GZ = MatrixXf::Zero(Nindiv, num_rhs);

    if (!gOpt.PRECOMPUTE_GRM) {
        // On-the-fly computation
        const size_t num_snps = gData.num_snps; 
        const float  inv_num_snp = (num_snps > 0) ? (1.0f / (float)num_snps) : 0.0f;
        const float  sig2g = solveData.sig2g;
        const size_t SNP_CHUNK_SIZE = 50000;  // TODO: user input

        size_t snp_start = 0;
        while (snp_start < num_snps)
        {
            size_t snp_end = std::min(snp_start + SNP_CHUNK_SIZE, num_snps);
            size_t chunk_size = snp_end - snp_start;

            // 2A) partial_x[s] = sum_{i=0..Nindiv-1}[ g_i^std(s) * Z(i,:) ]
            MatrixXf partial_x = MatrixXf::Zero(chunk_size, num_rhs);

            #pragma omp parallel
            {
                VectorXf local_acc(num_rhs);

                #pragma omp for schedule(dynamic)
                for (int s = (int)snp_start; s < (int)snp_end; ++s) {
                    int chunk_index = s - snp_start;
                    local_acc.setZero();

                    // accumulate local_acc = g_i^std(s) * Z(i,:)
                    for (int i = 0; i < Nindiv; i++) {
                        int genotype = getGenotype(s, i, cData, gData);
                        if (genotype < 0) genotype = 0; // missing => treat as 0
                        float maf = gData.allele_freqs[s];
                        float denom = 2.0f * maf * (1.0f - maf);
                        float inv_std = (denom > 1e-12f) ? (1.0f / std::sqrt(denom)) : 1.0f;
                        float g_std = (genotype - 2.0f * maf) * inv_std;
                        for (int col = 0; col < num_rhs; col++) {
                            local_acc[col] += g_std * Z(i, col);
                        }
                    }
                    for (int col = 0; col < num_rhs; col++) {
                        partial_x(chunk_index, col) = local_acc[col];
                    }
                }
            }

            // GZ(i,:) += sum_{s in chunk}[ g_i^std(s) * partial_x[s,:] ] * (sig2g * inv_num_snp)
            #pragma omp parallel
            {
                VectorXf local_res(num_rhs);

                #pragma omp for schedule(dynamic)
                for (int i = 0; i < Nindiv; i++) {
                    local_res.setZero();

                    for (int s = (int)snp_start; s < (int)snp_end; ++s) {
                        int chunk_index = s - snp_start;
                        int genotype = getGenotype(s, i, cData, gData);
                        if (genotype < 0) genotype = 0; // missing => treat as 0
                        float maf = gData.allele_freqs[s];
                        float denom = 2.0f * maf * (1.0f - maf);
                        float inv_std = (denom > 1e-12f) ? (1.0f / std::sqrt(denom)) : 1.0f;
                        float g_std = (genotype - 2.0f * maf) * inv_std;

                        for (int col = 0; col < num_rhs; col++) {
                            local_res[col] += g_std * partial_x(chunk_index, col);
                        }
                    }
                    for (int col = 0; col < num_rhs; col++) {
                        GZ(i, col) += sig2g * inv_num_snp * local_res[col];
                    }
                }
            }
            snp_start = snp_end;
        }
    } else {
        // precomputed GRM
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < Nindiv; i++) {
            for (int j = 0; j < Nindiv; j++) {
                float gij = gData.GRM_lt[ ltIndex(i,j) ];
                for (int col = 0; col < num_rhs; col++) {
                    GZ(i, col) += solveData.sig2g * gij * Z(j, col);
                }
            }
        }
    }

    if (multGRM_uhat) {
        result_matrix = GZ;
        return;
    }
    // Step 3: J * GZ:  result(r,:) += GZ[indiv[r],:]   (individual -> record space)
    for(int r = 0; r < Nrec; r++) {
        int i = static_cast<int>( cData.indiv(r) ); 
        for(int col = 0; col < num_rhs; col++) {
            result_matrix(r, col) += GZ(i, col);
        }
    }

    // ------------------------------------------------------------------
    // B) TIME PART: V_ij(block) = (b0 + b01*(t_i + t_j) + b1 * t_i*t_j)
    // ------------------------------------------------------------------
    const float sig2b0  = solveData.sig2b0;
    const float sigb01 = solveData.sigb01;
    const float sig2b1  = solveData.sig2b1;

    const auto& block_starts = solveData.block_starts; 
    const int num_blocks = (int)block_starts.size() - 1;

    #pragma omp parallel for
    for (int b = 0; b < num_blocks; b++) {
        int start = block_starts[b];
        int end   = block_starts[b+1];
        int block_size = end - start;

        for (int i = 0; i < block_size; i++) {
            float ti = cData.X(start + i, cData.p); // time of record
            for (int j = 0; j < block_size; j++) {
                float tj = cData.X(start + j, cData.p);
                float val = sig2b0 + sigb01*(ti + tj) + sig2b1*(ti * tj);

                for (int col = 0; col < num_rhs; col++) {
                    result_matrix(start + i, col) += val * input_matrix(start + j, col);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // C) RESIDUAL PART: add sig2e * X on the diagonal
    // ------------------------------------------------------------------
    const float sig2e = solveData.sig2e;
    #pragma omp parallel for
    for (int i = 0; i < Nrec; i++) {
        for (int col = 0; col < num_rhs; col++) {
            result_matrix(i, col) += sig2e * input_matrix(i, col);
        }
    }
}

// compute tildeV entry (like V, without GRM)
static inline float computeTildeV_ij(int i, int j,
                              const CohortData& cData,
                              const SolvingData& solveData)
{
    float val = 0.0f;
    int indiv_i = (int) cData.indiv(i);
    int indiv_j = (int) cData.indiv(j);

    if (indiv_i == indiv_j) {
        // JJ' => +1*sig2g if same individual
        val += solveData.sig2g;

        // UBU' => time-block
        float ti = cData.X(i, cData.p);
        float tj = cData.X(j, cData.p);
        val += (solveData.sig2b0 + solveData.sigb01*(ti + tj) + solveData.sig2b1*(ti * tj));
    }
    if (i == j) val += solveData.sig2e;
    return val;
}

// initializer for PCG solver (tildeV^{-1} * b)
void computeTildeVInv_b(const MatrixXf& b,
                        MatrixXf& x,
                        const CohortData& cData,
                        const SolvingData& solveData)
{
    const auto& block_starts = solveData.block_starts;
    int num_blocks = (int) block_starts.size() - 1;
    int num_rhs = b.cols();
    int Nrec = cData.Nrec;

    x.setZero(Nrec, num_rhs);

    #pragma omp parallel for
    for (int blk = 0; blk < num_blocks; blk++) {
        int start = block_starts[blk];
        int end   = block_starts[blk + 1];
        int block_size = end - start;

        MatrixXf tildeBlock(block_size, block_size);
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                int gi = start + i;
                int gj = start + j;
                tildeBlock(i, j) = computeTildeV_ij(gi, gj, cData, solveData);
            }
        }
        MatrixXf tildeBlock_inv = tildeBlock.inverse();
        MatrixXf b_block = b.block(start, 0, block_size, num_rhs);
        MatrixXf x_block = tildeBlock_inv * b_block;
        x.block(start, 0, block_size, num_rhs) = x_block;
    }
}

void runPCG_multi(const MatrixXf& b, 
                  MatrixXf& x,
                  const CohortData& cData,
                  const GenotypeData& gData,
                  SolvingData& solveData,
                  const GlobalOptions& gOpt)
{
    std::cout << "[runPCG_multi] Starting PCG solver..." << std::endl;
    
    const bool debug_mode = gOpt.DEBUG;
    int Nrec = cData.Nrec;
    int num_rhs = b.cols();
    float tol = solveData.PCG_TOL;
    int max_iter = solveData.PCG_MAX_ITER;

    // Build initial guess x0 = tildeV^{-1} * b
    // computeTildeVInv_b(b, x, cData, solveData);
    // MatrixXf Vx0(Nrec, num_rhs);
    // computeVx_multi(x, Vx0, cData, gData, solveData, gOpt, false);
    MatrixXf r = b; // - Vx0;

    MatrixXf z(Nrec, num_rhs);
    MatrixXf Vp(Nrec, num_rhs);
    VectorXf zcol(Nrec), rcol(Nrec); 

    // initial => z = Q^-1 r
    for (int j = 0; j < num_rhs; ++j) {
        rcol = r.col(j);
        applyPreconditioner(solveData, rcol, zcol);
        z.col(j) = zcol;
    }
    MatrixXf p = z;
    VectorXf rsold = (r.array() * z.array()).colwise().sum();
    
    // track residual
    float lastResidualRatio = 1e30f;
    {
        float initRatio = 0.f;
        for (int j = 0; j < num_rhs; j++) {
            float nr = r.col(j).norm();
            float nb = b.col(j).norm();
            if (nb > 1e-12f) {
                float ratio = nr / nb;
                if (ratio > initRatio) initRatio = ratio;
            }
        }
        lastResidualRatio = initRatio;
    }

    // stagnation detection (for cases of high number of records per individual)
    const int STAG_WINDOW = 10;
    const float STAG_FRAC = 5e-3f; 
    int stagCount = 0;

    for (int iter = 0; iter < max_iter; ++iter) {
        computeVx_multi(p, Vp, cData, gData, solveData, gOpt, false);

        VectorXf alpha = rsold.array() / (p.array() * Vp.array()).colwise().sum().transpose();
        
        for (int j = 0; j < num_rhs; ++j) {
            x.col(j).noalias() += alpha(j) * p.col(j);
            r.col(j).noalias() -= alpha(j) * Vp.col(j);
        }

        float maxRatio = 0.f;
        for (int j = 0; j < num_rhs; ++j) {
            float nr = r.col(j).norm();
            float nb = b.col(j).norm();
            float ratio = (nb > 1e-12f) ? (nr / nb) : 0.f;
            if (ratio > maxRatio) {
                maxRatio = ratio;
            }
        }
        if (maxRatio < tol) {
            std::cout << "[runPCG_multi] Converged at iteration " << iter + 1 << std::endl;
            break;
        }
        
        // Stagnation check
        float improvement = lastResidualRatio - maxRatio;
        float minRequired = STAG_FRAC * lastResidualRatio;
        if (improvement < minRequired) {
            stagCount++;
        } else {
            stagCount = 0;
        }
        lastResidualRatio = maxRatio;
        if (stagCount >= STAG_WINDOW) {
            std::cout << "[runPCG_multi] Stalled (no significant improvement for " << STAG_WINDOW << " iters)." << std::endl;
            std::cout << "[runPCG_multi] Converged at iteration " << iter + 1 << std::endl;
            break;
        }
        // if ((r.colwise().norm().array() < tol * b.colwise().norm().array()).all()) {
        //     std::cout << "[runPCG_multi] Converged at iteration " << iter + 1 << std::endl;
        //     break;
        // }
        for (int j = 0; j < num_rhs; ++j) {
            rcol = r.col(j);
            applyPreconditioner(solveData, rcol, zcol);
            z.col(j) = zcol;
        }
        VectorXf rsnew = (r.array() * z.array()).colwise().sum();
        VectorXf beta = rsnew.array() / rsold.array(); 
        rsold = rsnew; 
        
        for (int j = 0; j < num_rhs; ++j) {
            p.col(j).noalias() = z.col(j) + beta(j) * p.col(j);
        }

        // if (debug_mode) {
        //     std::cout << "[runPCG_multi] Iter " << iter+1 << " resid/b norms: [ ";
        //     for (int j = 0; j < num_rhs; ++j) {
        //         std::cout << r.col(j).norm() / b.col(j).norm() << " ";
        //     }
        //     std::cout << "]" << std::endl;
        // }
        if (debug_mode) {
            std::cout << "[runPCG_multi] Iter " << (iter+1) << " resid/b = " << maxRatio << " (stagCount = " << stagCount << ")\n";
        }
    }
}

// compute u_hat = sig2g * (A * J^T) V^-1 r
void compute_uhat(const VectorXf& Vinv_r, 
                  VectorXf& u_hat, 
                  const CohortData& cData, 
                  const GenotypeData& gData,
                  const SolvingData& solveData, 
                  const GlobalOptions& gOpt)
{
    const bool debug_mode = gOpt.DEBUG;
    int N = cData.N;
    int Nrec = cData.Nrec;
    u_hat.setZero(N);

    MatrixXf Vinv_r_mat = Vinv_r;
    MatrixXf temp(N, 1);
    computeVx_multi(Vinv_r_mat, temp, cData, gData, solveData, gOpt, /*multGRM_uhat=*/true);
    for (int i = 0; i < N; ++i) {
        u_hat[i] = temp(i, 0);
    }

    if(debug_mode) {
        std::cout << "[compute_uhat] First 5 elements of u_hat:\n";
        for (int i = 0; i < std::min(5, N); ++i) {
            std::cout << u_hat[i] << " ";
        }
        std::cout << std::endl;
    }
}

void compute_bhat(const VectorXf& Vinv_r, 
                  VectorXf& b_hat, 
                  const CohortData& cData,
                  const SolvingData& solveData, 
                  bool debug_mode)
{
    int N = cData.N;
    int Nrec = cData.Nrec;
    b_hat.setZero(2*N);
    const VectorXf indiv = cData.indiv;
    const VectorXf times = cData.X.col(cData.p);

    float sig2b0  = solveData.sig2b0;
    float sigb01 = solveData.sigb01;
    float sig2b1  = solveData.sig2b1;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < Nrec; ++i) {
        int indiv_id = indiv(i);
        float ti = times(i);

        float update1 = Vinv_r[i] * (sig2b0 + sigb01 * ti);
        float update2 = Vinv_r[i] * (sigb01 + sig2b1 * ti);

        #pragma omp critical
        {
            b_hat[2 * indiv_id] += update1;
            b_hat[2 * indiv_id + 1] += update2;
        }
    }
    
    if (debug_mode) {
        std::cout << "[compute_bhat] First 5 elements of b_hat:\n";
        for (int i = 0; i < std::min(5, 2*N); ++i) {
            std::cout << b_hat[i] << " ";
        }
        std::cout << std::endl;
    }
}
