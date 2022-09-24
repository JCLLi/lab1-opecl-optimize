// Copyright 2018 Delft University of Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "acsmatmult/matmult.h"
#include <omp.h>  // OpenMP support.

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")
/*************************************/

#define MAX_DIM 20000


Matrix<float> multiplyMatricesOMP(Matrix<float> a,
                                  Matrix<float> b,
                                  int num_threads) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE OPENMP HERE */

  //std::cout << "OpenMP test, brace for impact!" << std::endl;

/*
  // Test if OpenMP works:
#pragma omp parallel for
  for (int i = 0; i < 8; i++) {
    std::cout << "Hello World " << "from thread " << omp_get_thread_num() << std::endl;
  }

  return Matrix<float>(1, 1); */

  // Test if matrices can be multiplied.
  if (a.columns != b.rows) {
    throw std::domain_error("Matrix dimensions do not allow matrix-multiplication.");
  }

  // Height and width
  auto rows = a.rows;
  auto columns = b.columns;

  // Create the resulting matrix
  auto result = Matrix<float>(rows, columns);

  /* Set amount of threads */
  omp_set_num_threads(num_threads);

/*
#pragma omp parallel for
  for (size_t i = 0; i < rows; i++) {
 	for (size_t j = 0; j < columns; j++) {
		for (size_t k = 0; k < b.rows; k++) {
			result(i,j) += a(i,k)*b(k,j);
		}
	}
  }
*/

  //float flatA_float[MAX_DIM];
  //float flatB_float[MAX_DIM];
  float *flatA_float = (float*)malloc(sizeof(float)*rows*columns);
  float *flatB_float = (float*)malloc(sizeof(float)*rows*columns);

  /* Convert to 2d matrices */
  #pragma omp parallel for 
		for (size_t i = 0; i<rows; i++){
			for (size_t j=0; j<columns; j++) {
				flatA_float[i*rows+j] = a(i,j);
				flatB_float[j*columns+i] = b(i,j);
			}
		}

  size_t i, j, k;
  float tot;
  
  #pragma omp parallel shared(result) private(i,j,k,tot)
  {
	#pragma omp for schedule(static)
	  for (i=0; i<rows; i++) {
		for (j=0; j<columns;j++) {
			tot = 0.0;
			for (k=0; k<b.rows; k++) {
				tot += flatA_float[i*rows+k]*flatB_float[j*columns+k];
			}
			result(i,j) = tot;
		}
	  }
  }
  
  return result;
}

Matrix<double> multiplyMatricesOMP(Matrix<double> a,
                                   Matrix<double> b,
                                   int num_threads) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE OPENMP HERE */
  if (a.columns != b.rows) {
    throw std::domain_error("Matrix dimensions do not allow matrix-multiplication.");
  }

  // Height and width
  auto rows = a.rows;
  auto columns = b.columns;

  // Create the resulting matrix
  auto result = Matrix<double>(rows, columns);

  /* Set amount of threads */
  omp_set_num_threads(num_threads);

/*
#pragma omp parallel for
  for (size_t i = 0; i < rows; i++) {
 	for (size_t j = 0; j < columns; j++) {
		for (size_t k = 0; k < b.rows; k++) {
			result(i,j) += a(i,k)*b(k,j);
		}
	}
  }
*/

  //double flatA_double[MAX_DIM];
  //double flatB_double[MAX_DIM];
  double *flatA_double = (double*)malloc(sizeof(double)*rows*columns);
  double *flatB_double = (double*)malloc(sizeof(double)*rows*columns);

  /* Convert to 2d matrices */
  #pragma omp parallel for 
		for (size_t i = 0; i<rows; i++){
			for (size_t j=0; j<columns; j++) {
				flatA_double[i*rows+j] = a(i,j);
				flatB_double[j*columns+i] = b(i,j);
			}
		}

  size_t i, j, k;
  double tot;
  #pragma omp parallel shared(result) private(i,j,k,tot)
  {
	#pragma omp for schedule(static)
	  for (i=0; i<rows; i++) {
		for (j=0; j<columns;j++) {
			tot = 0.0;
			for (k=0; k<b.rows; k++) {
				tot += flatA_double[i*rows+k]*flatB_double[j*columns+k];
			}
			result(i,j) = tot;
		}
	  }
  }

  return result;
}
#pragma GCC pop_options
