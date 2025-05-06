#pragma warning(disable:4996)
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <CL/cl.h>
#include <iostream>
#include <chrono>
#include "layershape.h"
#include "cnnSeqFunc.h"
#include "cnn.h"

extern const char* CLASS_NAME[];

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

void print_images(float* images, int num_of_image, int nbyn) {
    for (int img = 0; img < num_of_image; img++) {
        printf("Image %d:\n", img + 1);
        // 각 이미지의 시작 포인터 계산
        float* image_ptr = images + img * nbyn * nbyn;

        for (int row = 0; row < nbyn; row++) {
            for (int col = 0; col < nbyn; col++) {
                // 현재 픽셀 값 접근
                float value = *(image_ptr + row * nbyn + col);
                printf("%6.2f ", value);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

char* get_source_code(const char* file_name, size_t* len) {
    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t length = (size_t)ftell(file);
    rewind(file);

    char* source_code = (char*)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';
    fclose(file);
    *len = length;

    return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char* log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    };
}

char* get_source_code(const char* file_name, size_t* len);
void build_error(cl_program program, cl_device_id device, cl_int err);

cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_command_queue queue_data;

size_t kernel_source_size;
cl_program program;

cl_kernel convt;
cl_kernel convt2;
cl_kernel maxPooling;
cl_kernel fcLayer;
cl_kernel convhelper;
cl_kernel convhighperf;
cl_kernel convhighperf_first;
cl_kernel reduction;



cl_mem images_buffer; // 추론할 입력 이미지 모두 들어가는 버퍼
cl_mem layer_buffer[21]; // 결과 레이어 버퍼
cl_mem w_buffer[21]; // 가중치 버퍼
cl_mem b_buffer[21]; // 편향 버퍼
cl_mem img2col_buffer[21];
cl_mem img2col_buffer2;

float* layer[21]; // 시작주소 계산용 변수
float* w[21];
float* b[21];


cl_event conv_help[20];


cl_ulong total_max_time = 0.0;
cl_ulong total_noTile_conv_time = 0.0;
cl_ulong total_wpt1_time = 0.0;
cl_ulong total_wpt2_time = 0.0;
cl_ulong total_fc_time = 0.0;
cl_mem temp_buffer;

void cnn_init(float* images, float* network, int num_of_image) {
    // Platform ID
    err = err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    // Device ID
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    // Create Context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    // Create Command Queue
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);
    queue_data = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    // Create Program Object
    size_t kernel_source_size;
    char* kernel_source = get_source_code("../../CNN_OPENCL/kernel.cl", &kernel_source_size);
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    // Build Program
    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    build_error(program, device, err);
    CHECK_ERROR(err);

    // create kernel object
    convt = clCreateKernel(program, "convt", &err);
    CHECK_ERROR(err);
    convt2 = clCreateKernel(program, "convt2", &err);
    CHECK_ERROR(err);
   
    convhighperf = clCreateKernel(program, "convhighperf", &err);
    CHECK_ERROR(err);
   
    convhelper = clCreateKernel(program, "convhelper", &err);
    CHECK_ERROR(err);
    maxPooling = clCreateKernel(program, "maxPooling", &err);
    CHECK_ERROR(err);
    fcLayer = clCreateKernel(program, "fcLayer", &err);
    CHECK_ERROR(err);
    convhighperf_first = clCreateKernel(program, "convhighperf_first", &err);
    CHECK_ERROR(err);
   
    

    int offset = 0;
    for (int i = 0; i < 17; ++i) {
        if (i == 2 || i == 5 || i == 9 || i == 13) i++;
        w[i] = network + offset;
        offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];
        b[i] = network + offset;
        offset += OUTPUT_DIM[i];
    }
    for (int i = 18; i < 21; ++i) {
        w[i] = network + offset;
        offset += INPUT_DIM[i] * OUTPUT_DIM[i];
        b[i] = network + offset;
        offset += OUTPUT_DIM[i];
    }

    for (int i = 0; i < 17; ++i) {
        if (i == 2 || i == 5 || i == 9 || i == 13) i++;	// pooling layer has no weights and biases
        w_buffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i], NULL, &err);
        CHECK_ERROR(err);
        b_buffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * OUTPUT_DIM[i], NULL, &err);
        CHECK_ERROR(err);
    }
    for (int i = 18; i < 21; ++i) {
        w_buffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * INPUT_DIM[i] * OUTPUT_DIM[i], NULL, &err);
        CHECK_ERROR(err);
        b_buffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * OUTPUT_DIM[i], NULL, &err);
        CHECK_ERROR(err);
    }

    for (int i = 0; i < 21; ++i) {
        layer[i] = (float*)calloc(OUTPUT_DIM[i] * NBYN[i] * NBYN[i] * num_of_image, sizeof(float));
        if (layer[i] == NULL) {
            perror("malloc error");
        }
    }

    for (int i = 0; i < 21; i++) {
        layer_buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * NBYN[i] * NBYN[i] * OUTPUT_DIM[i] * num_of_image, NULL, &err);
        CHECK_ERROR(err);
    }

    for (int i = 0; i < 21; i++) {
        img2col_buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 9 * INPUT_DIM[i] * NBYN[i] * NBYN[i] * num_of_image, NULL, &err);
        CHECK_ERROR(err);
    }
    //img2col_buffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 9 * INPUT_DIM[i] * NBYN[i] * NBYN[i] * num_of_image, NULL, &err);
    temp_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * NBYN[14] * NBYN[14] * OUTPUT_DIM[14] * 2 * num_of_image, NULL, &err);
   

}

void CONVHELPER(cl_mem input, int i, int num_of_image) {
    size_t GLOBAL_ITEM_SIZE[3] = {
        static_cast<size_t>(NBYN[i] * NBYN[i]),
        static_cast<size_t>(INPUT_DIM[i]),  
        static_cast<size_t>(num_of_image)
    };

    size_t LOCAL_ITEM_SIZE[3] = {
        static_cast<size_t>(NBYN[i] * NBYN[i]),
        static_cast<size_t>(1),
        static_cast<size_t>(1)
    };

    err = clSetKernelArg(convhelper, 0, sizeof(cl_mem), &input);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhelper, 1, sizeof(int), &NBYN[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhelper, 2, sizeof(cl_mem), &img2col_buffer[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhelper, 3, sizeof(float) * NBYN[i] * NBYN[i], NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhelper, 4, sizeof(int), &INPUT_DIM[i]);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, convhelper, 3, NULL, GLOBAL_ITEM_SIZE, LOCAL_ITEM_SIZE, 0, NULL, NULL);
    CHECK_ERROR(err);
}



void CONVT(cl_mem input, cl_mem output, int i, int num_of_image) {
    int WPT = 16;
    int ts = 32;
    int img_num_per_ch = 64;
    if (num_of_image == 184)
        img_num_per_ch = 184;
    
    size_t GLOBAL_ITEM_SIZE[3] = {
        static_cast<size_t>(NBYN[i] * NBYN[i]* img_num_per_ch),
        static_cast<size_t>(OUTPUT_DIM[i]/16),
        static_cast<size_t>(num_of_image/ (img_num_per_ch))
    };

    size_t LOCAL_ITEM_SIZE[3] = {
        static_cast<size_t>(ts),
        static_cast<size_t>(ts/16),
        static_cast<size_t>(1)
    };

    err = clSetKernelArg(convt, 0, sizeof(cl_mem), &input);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt, 1, sizeof(cl_mem), &w_buffer[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt, 2, sizeof(cl_mem), &b_buffer[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt, 3, sizeof(cl_mem), &output);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt, 4, sizeof(int), &INPUT_DIM[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt, 5, sizeof(int), &OUTPUT_DIM[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt, 6, sizeof(int), &NBYN[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt, 7, sizeof(float) * ts * ts, NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt, 8, sizeof(float) * ts * ts, NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt, 9, sizeof(int), &ts);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt, 10, sizeof(int), &img_num_per_ch);
    CHECK_ERROR(err);
    
    err = clEnqueueNDRangeKernel(queue, convt, 3, NULL, GLOBAL_ITEM_SIZE, LOCAL_ITEM_SIZE, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void CONV_HIGHPERF(cl_mem input, cl_mem output, int i, int num_of_image) {
    size_t GLOBAL_ITEM_SIZE[3] = {
        static_cast<size_t>(NBYN[i] * NBYN[i]),
        static_cast<size_t>(OUTPUT_DIM[i] / 16),
        static_cast<size_t>(num_of_image)
    };

    size_t LOCAL_ITEM_SIZE[3] = {
        static_cast<size_t>(TS[i]),
        static_cast<size_t>(TS[i] / 16),
        static_cast<size_t>(1)
    };

    err = clSetKernelArg(convhighperf, 0, sizeof(cl_mem), &input);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf, 1, sizeof(cl_mem), &w_buffer[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf, 2, sizeof(cl_mem), &b_buffer[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf, 3, sizeof(cl_mem), &output);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf, 4, sizeof(int), &INPUT_DIM[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf, 5, sizeof(int), &OUTPUT_DIM[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf, 6, sizeof(int), &NBYN[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf, 7, sizeof(float) * TS[i] * TS[i], NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf, 8, sizeof(float) * TS[i] * TS[i], NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf, 9, sizeof(int), &TS[i]);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, convhighperf, 3, NULL, GLOBAL_ITEM_SIZE, LOCAL_ITEM_SIZE, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void CONV_HIGHPERF_FIRST(cl_mem input, cl_mem output, int i, int num_of_image) {
    size_t GLOBAL_ITEM_SIZE[3] = {
        static_cast<size_t>(NBYN[i] * NBYN[i]),
        static_cast<size_t>(OUTPUT_DIM[i] / 16),
        static_cast<size_t>(num_of_image)
    };

    size_t LOCAL_ITEM_SIZE[3] = {
        static_cast<size_t>(TS[i]),
        static_cast<size_t>(TS[i] / 16),
        static_cast<size_t>(1)
    };

    err = clSetKernelArg(convhighperf_first, 0, sizeof(cl_mem), &input);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf_first, 1, sizeof(cl_mem), &w_buffer[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf_first, 2, sizeof(cl_mem), &b_buffer[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf_first, 3, sizeof(cl_mem), &output);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf_first, 4, sizeof(int), &INPUT_DIM[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf_first, 5, sizeof(int), &OUTPUT_DIM[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf_first, 6, sizeof(int), &NBYN[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf_first, 7, sizeof(float) * TS[i] * TS[i], NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf_first, 8, sizeof(float) * TS[i] * TS[i], NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(convhighperf_first, 9, sizeof(int), &TS[i]);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, convhighperf_first, 3, NULL, GLOBAL_ITEM_SIZE, LOCAL_ITEM_SIZE, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void CONT2(cl_mem input, cl_mem output, int i, int num_of_image) {
    int WPT = 16;
    int ts = 64;
    int img_num_per_ch = 8;
    if (num_of_image == 184)
        img_num_per_ch = 184;

    size_t GLOBAL_ITEM_SIZE[3] = {
        static_cast<size_t>(NBYN[i] * NBYN[i] * img_num_per_ch),
        static_cast<size_t>(OUTPUT_DIM[i] / 16),
        static_cast<size_t>(num_of_image / (img_num_per_ch))
    };

    size_t LOCAL_ITEM_SIZE[3] = {
        static_cast<size_t>(ts),
        static_cast<size_t>(ts / 16),
        static_cast<size_t>(1)
    };

    err = clSetKernelArg(convt2, 0, sizeof(cl_mem), &input);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt2, 1, sizeof(cl_mem), &w_buffer[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt2, 2, sizeof(cl_mem), &b_buffer[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt2, 3, sizeof(cl_mem), &output);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt2, 4, sizeof(int), &INPUT_DIM[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt2, 5, sizeof(int), &OUTPUT_DIM[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt2, 6, sizeof(int), &NBYN[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt2, 7, sizeof(float) * ts * ts, NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt2, 8, sizeof(float) * ts * ts, NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt2, 9, sizeof(int), &ts);
    CHECK_ERROR(err);
    err = clSetKernelArg(convt2, 10, sizeof(int), &img_num_per_ch);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, convt2, 3, NULL, GLOBAL_ITEM_SIZE, LOCAL_ITEM_SIZE, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void MAXPOOLING(cl_mem input, cl_mem output, int i, int num_of_image) {
    size_t GLOBAL_ITEM_SIZE[3] = {
        static_cast<size_t>(NBYN[i] * NBYN[i]),
        static_cast<size_t>(INPUT_DIM[i]),
        static_cast<size_t>(num_of_image)
    };

    size_t LOCAL_ITEM_SIZE[3] = {
        static_cast<size_t>(NBYN[i] * NBYN[i]),
        static_cast<size_t>(1),
        static_cast<size_t>(1)
    };

    err = clSetKernelArg(maxPooling, 0, sizeof(cl_mem), &input);
    CHECK_ERROR(err);
    err = clSetKernelArg(maxPooling, 1, sizeof(cl_mem), &output);
    CHECK_ERROR(err);
    err = clSetKernelArg(maxPooling, 2, sizeof(int), &INPUT_DIM[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(maxPooling, 3, sizeof(int), &NBYN[i]);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, maxPooling, 3, NULL, GLOBAL_ITEM_SIZE, LOCAL_ITEM_SIZE, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void FCLAYER(cl_mem input, cl_mem output, int i, int num_of_image) {
    size_t GLOBAL_ITEM_SIZE[3] = {
        static_cast<size_t>(OUTPUT_DIM[i]),
        static_cast<size_t>(INPUT_DIM[i]),
        static_cast<size_t>(num_of_image)
    };

    size_t LOCAL_ITEM_SIZE[3] = {
        1, INPUT_DIM[i], 1
    };

    err = clSetKernelArg(fcLayer, 0, sizeof(cl_mem), &input);
    CHECK_ERROR(err);
    err = clSetKernelArg(fcLayer, 1, sizeof(cl_mem), &output);
    CHECK_ERROR(err);
    err = clSetKernelArg(fcLayer, 2, sizeof(cl_mem), &w_buffer[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(fcLayer, 3, sizeof(cl_mem), &b_buffer[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(fcLayer, 4, sizeof(int), &INPUT_DIM[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(fcLayer, 5, sizeof(int), &OUTPUT_DIM[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(fcLayer, 6, sizeof(int), &NBYN[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(fcLayer, 7, sizeof(float) * INPUT_DIM[i], NULL);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, fcLayer, 3, NULL, GLOBAL_ITEM_SIZE, LOCAL_ITEM_SIZE, 0, NULL, NULL);
    CHECK_ERROR(err);
}

void cnn(float* images, float* network, int* labels, float* confidences, int num_of_image) {
    int batch = 256;
    
    cnn_init(images, network, batch);

    images_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 32 * 32 * INPUT_DIM[0] * batch, NULL, &err);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 17; ++i) {
        if (i == 2 || i == 5 || i == 9 || i == 13) i++;
        err = clEnqueueWriteBuffer(queue, w_buffer[i], CL_TRUE, 0, sizeof(float) * 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i], w[i], 0, NULL, NULL);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue, b_buffer[i], CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[i], b[i], 0, NULL, NULL);
        CHECK_ERROR(err);
    }

    for (int i = 18; i < 21; ++i) {
        err = clEnqueueWriteBuffer(queue, w_buffer[i], CL_TRUE, 0, sizeof(float) * INPUT_DIM[i] * OUTPUT_DIM[i], w[i], 0, NULL, NULL);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue, b_buffer[i], CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[i], b[i], 0, NULL, NULL);
        CHECK_ERROR(err);
    }
   

    for (int i = 0; i < num_of_image; i = i + batch) {
        if (i == batch * (3000 / batch)) {
            batch = 3000 - i;
        }
        err = clEnqueueWriteBuffer(queue, images_buffer, CL_FALSE, 0, sizeof(float) * 32 * 32 * INPUT_DIM[0] * batch, images, 0, NULL, NULL);
        CHECK_ERROR(err);

        CONVHELPER(images_buffer, 0, batch);
        CONV_HIGHPERF_FIRST(img2col_buffer[0], layer_buffer[0], 0, batch);
        CONVHELPER(layer_buffer[0], 1, batch);
        CONV_HIGHPERF(img2col_buffer[1], layer_buffer[1], 1, batch);
        MAXPOOLING(layer_buffer[1], layer_buffer[2], 2, batch);

        CONVHELPER(layer_buffer[2], 3, batch);
        CONV_HIGHPERF(img2col_buffer[3], layer_buffer[3], 3, batch);
        CONVHELPER(layer_buffer[3], 4, batch);
        CONV_HIGHPERF(img2col_buffer[4], layer_buffer[4], 4, batch);
        MAXPOOLING(layer_buffer[4], layer_buffer[5], 5, batch);

        CONVHELPER(layer_buffer[5], 6, batch);
        CONV_HIGHPERF(img2col_buffer[6], layer_buffer[6], 6, batch);
        CONVHELPER(layer_buffer[6], 7, batch);
        CONV_HIGHPERF(img2col_buffer[7], layer_buffer[7], 7, batch);
        CONVHELPER(layer_buffer[7], 8, batch);
        CONV_HIGHPERF(img2col_buffer[8], layer_buffer[8], 8, batch);
        MAXPOOLING(layer_buffer[8], layer_buffer[9], 9, batch);

        CONVHELPER(layer_buffer[9], 10, batch);
        CONT2(img2col_buffer[10], layer_buffer[10], 10, batch);
        CONVHELPER(layer_buffer[10], 11, batch);
        CONT2(img2col_buffer[11], layer_buffer[11], 11, batch);
        CONVHELPER(layer_buffer[11], 12, batch);
        CONT2(img2col_buffer[12], layer_buffer[12], 12, batch);
        MAXPOOLING(layer_buffer[12], layer_buffer[13], 13, batch);

        CONVHELPER(layer_buffer[13], 14, batch);
        CONVT(img2col_buffer[14], layer_buffer[14], 14, batch);
        CONVHELPER(layer_buffer[14], 15, batch);
        CONVT(img2col_buffer[15], layer_buffer[15], 15, batch);
        CONVHELPER(layer_buffer[15], 16, batch);
        CONVT(img2col_buffer[16], layer_buffer[16], 16, batch);
        MAXPOOLING(layer_buffer[16], layer_buffer[17], 17, batch);

        FCLAYER(layer_buffer[17], layer_buffer[18], 18, batch);
        FCLAYER(layer_buffer[18], layer_buffer[19], 19, batch);
        FCLAYER(layer_buffer[19], layer_buffer[20], 20, batch);

        err = clEnqueueReadBuffer(queue, layer_buffer[20], CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[20] * NBYN[20] * NBYN[20] * batch, layer[20], 0, NULL, NULL);
        CHECK_ERROR(err);

        for (int k = 0; k < batch; k++) {
            softmax(layer[20] + 10 * k, 10);
            labels[i + k] = find_max(layer[20] + 10 * k, 10);
            confidences[i + k] = layer[20][10 * k + labels[i + k]];
        }

        images += 32 * 32 * 3 * batch;
    }

     auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> execution_time_multi = end - start;
    std::cout << "Execution time: " << execution_time_multi.count() << " sec" << std::endl;
    
   
    
}