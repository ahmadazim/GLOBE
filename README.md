# GLOBE

**G**enomic **L**ongitudinal **O**utcomes with **B**iobank **E**nrichment (**GLOBE**) is a tool (implemented in C/C++) for solving large-scale longitudinal genomic mixed models. It leverages repeated phenotype records over time alongside genetic data, enabling more accurate and interpretable polygenic risk prediction and progression assessment for complex diseases. Iterative, memory-efficient strategies are leveraged throughout, ensuring that biobank-scale analyses remain computationally feasible.

## Key Features
- Efficient iterative solver for Mixed Model Equations (MMEs) and variance component estimation (AIREML).
- Highly memory efficient: avoids storing large matrices, and instead uses on-the-fly computations and preconditioning.
- Scales to large cohorts and repeated measures (feasible for millions of records and millions of SNPs).
- Fast parallelization via OpenMP.

## Compilation
1. Clone or download this repository:
   ```bash
   git clone https://github.com/ahmadazim/GLOBE.git
   cd GLOBE
   ```
2. Create and enter a build directory:
    ```bash
    mkdir build && cd build
    ```
3. Configure with CMake:
    ```bash
    cmake ..
    ```
4. Compile: 
    ```bash
    make -j
    ```

## Executables
After compiling, two main executables are created in the `build/` directory:


### 1. **runPCGsolver**

**Purpose:** Fits a longitudinal genomic mixed model (LGMM) using user-specified variance components.

**Usage:**
 ```bash
 ./build/runPCGsolver \
     --input <file> \
     --output <prefix> \
     --sig2e <val> --sig2g <val> --sig2b0 <val> --sigb01 <val> --sig2b1 <val> \
     [--geno <plinkPrefix> | --grm <grmPrefix>] \
     [--debug] [--minMAF <val>] [--maxMissingRate <val>]
 ```

**Example:**
 ```bash 
 ./build/runPCGsolver \
     --input data.txt \
     --output results \
     --geno /path/to/myPlinkFile \
     --sig2e 0.6 --sig2g 0.4 --sig2b0 0.4 --sigb01 0.2 --sig2b1 0.2
 ```


### 2. **runAIREML**

**Purpose:** Estimates variance components for LGMM equations using AIREML.

**Usage:**
 ```bash
 ./build/runAIREML \
     <input_filename> <grm_prefix> <output_prefix> \
     [--maxiter <val>] [--tol <val>] [--numMC <val>] \
     [--sig2e <val>] [--sig2g <val>] [--sig2b0 <val>] [--sigb01 <val>] [--sig2b1 <val>]
 ```

**Example:**
 ```bash 
 ./build/runAIREML \
     data.txt \
     mydata.grm.bin \
     output_results \
     --maxiter 50 --tol 1e-3 --numMC 15
 ```

## License

This package is available under the MIT license.

## Contact

For questions, please reach out via GitHub issues or open a pull request.
