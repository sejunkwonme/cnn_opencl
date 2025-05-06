#pragma once
#ifndef _CNN_H
#define _CNN_H
void cnn_seq(float* images, float* network, int* labels, float* confidences, int num_of_image);
void compare(const char* filename, int num_of_image);

void cnn(float* images, float* network, int* labels_opencl, float* confidences_opencl, int num_of_image);
#endif