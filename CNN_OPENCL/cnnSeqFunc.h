#pragma once
#ifndef _CNNSEQFUNC_H
#define _CNNSEQFUNC_H
#include <cstring>
#include <math.h>
static void convolution(float* inputs, float* outputs, float* filter, float* biases, int inDim, int outDim, int nbyn) {
	memset(outputs, 0, nbyn * nbyn * outDim * sizeof(float));
	int x = 0, y = 0;
	int offset = nbyn * nbyn;
	float sum = 0, temp;
	float* input, * output;

	for (int outNeuron = 0; outNeuron < outDim; ++outNeuron) {
		input = inputs;
		for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
			output = outputs;
			for (int row = 0; row < nbyn; ++row) {
				for (int col = 0; col < nbyn; ++col) {
					sum = 0;
					for (int fRow = 0; fRow < 3; ++fRow) {
						for (int fCol = 0; fCol < 3; ++fCol) {
							x = col + fCol - 1;
							y = row + fRow - 1;

							if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
								sum += input[nbyn * y + x] * filter[3 * fRow + fCol];
							}

						}
					}
					*(output++) += sum;
				}
			}
			filter += 9;
			input += offset;

		}
		for (int i = 0; i < offset; ++i) {
			(*outputs) = (*outputs) + (*biases);
			if (*outputs < 0) (*outputs) = 0;	//ReLU
			outputs++;
		}
		++biases;
	}
}

static void max_pooling(float* input, float* output, int DIM, int nbyn) {
	float max, temp;
	int n, row, col, x, y;
	for (n = 0; n < DIM; ++n) {
		for (row = 0; row < nbyn; row += 2) {
			for (col = 0; col < nbyn; col += 2) {
				//max = -FLT_MAX;
				max = 0;
				for (y = 0; y < 2; ++y) {
					for (x = 0; x < 2; ++x) {
						temp = input[nbyn * (row + y) + col + x];
						if (max < temp) max = temp;
					}
				}
				*(output++) = max;
			}
		}
		input += nbyn * nbyn;
	}
}

void fc_layer(float* input, float* output, float* weights, float* biases, int inDim, int outDim) {
	float sum;
	for (int outNeuron = 0; outNeuron < outDim; ++outNeuron) {
		sum = 0;
		for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
			sum += input[inNeuron] * (*weights++);
		}
		sum += biases[outNeuron];
		if (sum > 0) output[outNeuron] = sum;	//ReLU
		else output[outNeuron] = 0;
	}
}

static void softmax(float* input, int N) {
	int i;
	float* iter;
	float max = *input; // 각 부분의 첫번째값
	for (iter = input; iter < input + N; iter++) {
		if (max < *iter) max = *iter;
	}
	float sum = 0;
	for (iter = input; iter < input + N; iter++) {
		sum += exp(*iter - max);
	}
	for (iter = input; iter < input + N; iter++) {
		*iter = exp(*iter - max) / (sum + 1e-7);
	}
}

static int find_max(float* input, int classNum) {
	if (input == NULL || classNum <= 0) {
		// 유효하지 않은 입력 처리
		return -1;
	}

	int maxIndex = 0;
	float max = *input;
	float* iter = input;
	for (int i = 1; i < classNum; i++) {
		iter++;
		if (*iter > max) {
			max = *iter;
			maxIndex = i;
		}
	}
	return maxIndex;
}

#endif