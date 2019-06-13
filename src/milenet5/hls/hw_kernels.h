
#ifndef _HW_KERNELS_H
#define _HW_KERNELS_H

#include "param.h"


#pragma SDS data mem_attribute(input:PHYSICAL_CONTIGUOUS,output:PHYSICAL_CONTIGUOUS,weight:PHYSICAL_CONTIGUOUS,bias:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(input:SEQUENTIAL, output:SEQUENTIAL, weight:SEQUENTIAL, bias:SEQUENTIAL)
#pragma SDS data zero_copy(input, weight, bias, output)
void convolution_1hw(decimal_t input[INPUT_AMT * INPUT_SZ * INPUT_SZ], decimal_t output[LAYER1_AMT * LAYER1_SZ * LAYER1_SZ],
		decimal_t weight[INPUT_AMT * LAYER1_AMT * KERNEL_SZ * KERNEL_SZ], decimal_t bias[LAYER1_AMT], int allocate);

#pragma SDS data mem_attribute(input:PHYSICAL_CONTIGUOUS,output:PHYSICAL_CONTIGUOUS,weight:PHYSICAL_CONTIGUOUS,bias:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(input:SEQUENTIAL, output:SEQUENTIAL, weight:SEQUENTIAL, bias:SEQUENTIAL)
#pragma SDS data zero_copy(input, weight, bias, output)
void convolution_2hw(decimal_t input[LAYER2_AMT * LAYER2_SZ * LAYER2_SZ], decimal_t output[LAYER3_AMT * LAYER3_SZ * LAYER3_SZ],
		decimal_t weight[LAYER2_AMT * LAYER3_AMT * KERNEL_SZ * KERNEL_SZ], decimal_t bias[LAYER3_AMT], int allocate);

#pragma SDS data mem_attribute(input:PHYSICAL_CONTIGUOUS,output:PHYSICAL_CONTIGUOUS,weight:PHYSICAL_CONTIGUOUS,bias:PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern(input:SEQUENTIAL, output:SEQUENTIAL, weight:SEQUENTIAL, bias:SEQUENTIAL)
#pragma SDS data zero_copy(input, weight, bias, output)
void convolution_3hw(decimal_t input[LAYER4_AMT * LAYER4_SZ * LAYER4_SZ], decimal_t output[LAYER5_AMT * LAYER5_SZ * LAYER5_SZ],
		decimal_t weight[LAYER4_AMT * LAYER5_AMT * KERNEL_SZ * KERNEL_SZ], decimal_t bias[LAYER5_AMT], int allocate);

#endif
