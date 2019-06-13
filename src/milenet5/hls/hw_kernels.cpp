#include "hw_kernels.h"


decimal_t relu_hw(decimal_t x) {
	#pragma HLS inline
	return x > 0 ? x : (decimal_t) 0.0;
}


void convolution_1hw(decimal_t input[INPUT_AMT * INPUT_SZ * INPUT_SZ], decimal_t output[LAYER1_AMT * LAYER1_SZ * LAYER1_SZ],
		decimal_t weight[INPUT_AMT * LAYER1_AMT * KERNEL_SZ * KERNEL_SZ], decimal_t bias[LAYER1_AMT], int allocate) {

	decimal_t i_bram[INPUT_SZ][INPUT_SZ];
	decimal_t w_bram[LAYER1_AMT][KERNEL_SZ][KERNEL_SZ];
	decimal_t b_bram[LAYER1_AMT];
	decimal_t o_bram[LAYER1_AMT][LAYER1_SZ][LAYER1_SZ];

	#pragma HLS array_partition variable=w_bram complete dim=1
	#pragma HLS array_partition variable=o_bram complete dim=1

	if (allocate) {
		for (int l = 0; l < LAYER1_AMT; l++)
			for (int k1 = 0; k1 < KERNEL_SZ; k1++)
				for (int k2 = 0; k2 < KERNEL_SZ; k2++)
					#pragma HLS PIPELINE II=1
					w_bram[l][k1][k2] = weight[k2 + KERNEL_SZ * (k1 + KERNEL_SZ * l)];

		for (int l = 0; l < LAYER1_AMT; l++)
			#pragma HLS PIPELINE II=1
			b_bram[l] = bias[l];
	}

	for (int l = 0; l < LAYER1_AMT; l++)
		for (int y = 0; y < LAYER1_SZ; y++)
			for (int x = 0; x < LAYER1_SZ; x++)
				#pragma HLS pipeline II=1
				o_bram[l][y][x] = 0;


	for (int y = 0; y < INPUT_SZ; y++)
		for (int x = 0; x < INPUT_SZ; x++)
			#pragma HLS PIPELINE II=1
			i_bram[y][x] = input[y * INPUT_SZ + x];

	for (int k1 = 0; k1 < KERNEL_SZ; k1++)
	for (int k2 = 0; k2 < KERNEL_SZ; k2++)
		for (int y = 0; y < LAYER1_SZ; y++)
		for (int x = 0; x < LAYER1_SZ; x++) {

			#pragma HLS PIPELINE II=1
			decimal_t in = i_bram[y+k1][x+k2];
			for (int l = 0; l < LAYER1_AMT; l++)
				#pragma HLS unroll
				o_bram[l][y][x] += in * w_bram[l][k1][k2];
		}

	for (int l = 0; l < LAYER1_AMT; l++)
		for (int y = 0; y < LAYER1_SZ; y++)
			for (int x = 0; x < LAYER1_SZ; x++)
				#pragma HLS PIPELINE II=1
				output[x + LAYER1_SZ * (y + l * LAYER1_SZ)] = relu_hw(o_bram[l][y][x] + b_bram[l]);


}



void convolution_2hw(decimal_t input[LAYER2_AMT * LAYER2_SZ * LAYER2_SZ], decimal_t output[LAYER3_AMT * LAYER3_SZ * LAYER3_SZ],
		decimal_t weight[LAYER2_AMT * LAYER3_AMT * KERNEL_SZ * KERNEL_SZ], decimal_t bias[LAYER3_AMT], int allocate) {

	decimal_t i_bram[LAYER2_AMT][LAYER2_SZ][LAYER2_SZ];
	decimal_t w_bram[LAYER2_AMT][LAYER3_AMT][KERNEL_SZ][KERNEL_SZ];
	decimal_t b_bram[LAYER3_AMT];
	decimal_t o_bram[LAYER3_AMT][LAYER3_SZ][LAYER3_SZ];

	#pragma HLS array_partition variable=w_bram complete dim=2
	#pragma HLS array_partition variable=o_bram complete dim=1

	if (allocate) {
		for (int l2 = 0; l2 < LAYER2_AMT; l2++)
			for (int l1 = 0; l1 < LAYER3_AMT; l1++)
				for (int k1 = 0; k1 < KERNEL_SZ; k1++)
					for (int k2 = 0; k2 < KERNEL_SZ; k2++)
						#pragma HLS PIPELINE II=1
						w_bram[l2][l1][k1][k2] = weight[k2 + KERNEL_SZ * (k1 + KERNEL_SZ * (l1 + l2 * LAYER3_AMT))];
		for (int l = 0; l < LAYER3_AMT; l++)
			#pragma HLS PIPELINE II=1
			b_bram[l] = bias[l];
	}

	for (int l = 0; l < LAYER2_AMT; l++)
		for (int y = 0; y < LAYER2_SZ; y++)
			for (int x = 0; x < LAYER2_SZ; x++)
				#pragma HLS PIPELINE II=1
				i_bram[l][y][x] = input[x + LAYER2_SZ * (y + LAYER2_SZ * l)];


	for (int l1 = 0; l1 < LAYER3_AMT; l1++)
		for (int y = 0; y < LAYER3_SZ; y++)
			for (int x = 0; x < LAYER3_SZ; x++)
				#pragma HLS pipeline II=1
				o_bram[l1][y][x] = 0;

	for (int k1 = 0; k1 < KERNEL_SZ; k1++)
	for (int k2 = 0; k2 < KERNEL_SZ; k2++)
	for (int l2 = 0; l2 < LAYER2_AMT; l2++)
		for (int y = 0; y < LAYER3_SZ; y++)
			for (int x = 0; x < LAYER3_SZ; x++) {

					#pragma HLS PIPELINE II=1
					decimal_t in = i_bram[l2][y+k1][x+k2];
					for (int l1 = 0; l1 < LAYER3_AMT; l1++)
						#pragma HLS unroll
						o_bram[l1][y][x] += in * w_bram[l2][l1][k1][k2];
				}

	for (int l = 0; l < LAYER3_AMT; l++)
		for (int y = 0; y < LAYER3_SZ; y++)
			for (int x = 0; x < LAYER3_SZ; x++)
				#pragma HLS PIPELINE II=1
				output[x + LAYER3_SZ * (y + l * LAYER3_SZ)] = relu_hw(o_bram[l][y][x] + b_bram[l]);


}



void convolution_3hw(decimal_t input[LAYER4_AMT * LAYER4_SZ * LAYER4_SZ], decimal_t output[LAYER5_AMT * LAYER5_SZ * LAYER5_SZ],
		decimal_t weight[LAYER4_AMT * LAYER5_AMT * KERNEL_SZ * KERNEL_SZ], decimal_t bias[LAYER5_AMT], int allocate) {

	decimal_t i_bram[LAYER4_AMT][LAYER4_SZ][LAYER4_SZ];
	decimal_t w_bram[LAYER5_AMT][LAYER4_AMT][KERNEL_SZ][KERNEL_SZ];
	decimal_t b_bram[LAYER5_AMT];
	decimal_t o_bram[LAYER5_AMT];

	#pragma HLS array_partition variable=w_bram complete dim=2
	#pragma HLS array_partition variable=i_bram complete dim=1

	if (allocate) {
		for (int l1 = 0; l1 < LAYER5_AMT; l1++)
			for (int l2 = 0; l2 < LAYER4_AMT; l2++)
				for (int k1 = 0; k1 < KERNEL_SZ; k1++)
					for (int k2 = 0; k2 < KERNEL_SZ; k2++)
						#pragma HLS PIPELINE II=1
						w_bram[l1][l2][k1][k2] = weight[k2 + KERNEL_SZ * (k1 + KERNEL_SZ * (l1 + l2 * LAYER5_AMT))];

		for (int l = 0; l < LAYER5_AMT; l++)
			#pragma HLS PIPELINE II=1
			b_bram[l] = bias[l];
	}

	for (int l = 0; l < LAYER4_AMT; l++)
		for (int y = 0; y < LAYER4_SZ; y++)
			for (int x = 0; x < LAYER4_SZ; x++)
				#pragma HLS PIPELINE II=1
				i_bram[l][y][x] = input[x + LAYER4_SZ * (y + LAYER4_SZ * l)];


	for (int l = 0; l < LAYER5_AMT; l++)
		#pragma HLS pipeline II=1
		o_bram[l] = 0;

	const int P = 4;


	for (int k1 = 0; k1 < KERNEL_SZ; k1++)
	for (int k2 = 0; k2 < KERNEL_SZ; k2++)
	for (int l1 = 0; l1 < LAYER5_AMT; l1++) {

		#pragma HLS PIPELINE II=1

		decimal_t mult[LAYER4_AMT];
		#pragma HLS array_partition variable=mult complete dim=1

		for (int l2 = 0; l2 < LAYER4_AMT; l2++)
			#pragma HLS UNROLL
			mult[l2] = i_bram[l2][k1][k2] * w_bram[l1][l2][k1][k2];

		for (int p = 0; p < P; p++)
			#pragma HLS UNROLL
			o_bram[l1] +=  mult[4*p] + mult[4*p+1] + mult[4*p+2] + mult[4*p+3];
	}

	for (int l = 0; l < LAYER5_AMT; l++)
		#pragma HLS PIPELINE II=1
		output[l] = relu_hw(o_bram[l] + b_bram[l]);
}



