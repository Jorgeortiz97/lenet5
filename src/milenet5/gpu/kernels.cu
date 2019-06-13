
//////////////////////////////////////
// KERNELS PARA ACELERAR CON LA GPU //
//////////////////////////////////////

__global__
void input_kernel(float *d_out, unsigned char *d_in, int LO_SZ, int mean, int std) {
	
	int x = threadIdx.x, y = blockIdx.x;

	int isInBorder = (x < PADDING || x >= IMG_SZ + PADDING || y < PADDING || y >= IMG_SZ + PADDING);
	

	int i = x - PADDING + IMG_SZ * (y - PADDING);	
	int o = x + LO_SZ * y;

	d_out[o] = isInBorder ? 0.0f : (float) ((int) (d_in[i] - mean) / std);
	
}

template <int LI, int LI_SZ> __global__
void conv_kernel(float *d_out, float *d_in, float *d_weight, float *d_bias, int LO, int LO_SZ) {
	
	__shared__ float s_input[LI][LI_SZ][LI_SZ];
	
	int x = threadIdx.x, y = threadIdx.y, z = blockIdx.x;

	for (int l1 = 0; l1 < LI; l1++)
		s_input[l1][y][x] = d_in[x + LI_SZ * (y + LI_SZ * l1)];
	
	if (x >= LO_SZ || y >= LO_SZ)
		return;
	
	__syncthreads();
	
	float value = d_bias[z];
	
	for (int l1 = 0; l1 < LI; l1++)
		#pragma unroll
		for (int k1 = 0; k1 < KERNEL_SZ; k1++)
		for (int k2 = 0; k2 < KERNEL_SZ; k2++)
			value += s_input[l1][y + k1][x + k2] * d_weight[k2 + KERNEL_SZ * (k1 + KERNEL_SZ * (z + l1 * LO))];

	d_out[x + LO_SZ * (y + z * LO_SZ)] = (value > 0) ? value : 0;

}


__global__
void subsamp_kernel(float *d_out, float *d_in, int dim, int LI_SZ, int LO, int LO_SZ) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= LO_SZ || y >= LO_SZ || z >= LO) return;

    int i, o = x + LO_SZ * (y + LO_SZ * z);
    float max_val = -FLT_MAX;

    for (int l1 = 0; l1 < dim; l1++)
    for (int l2 = 0; l2 < dim; l2++) {
        i = x * dim + l2 + LI_SZ * (y * dim + l1 + LI_SZ * z);
	    if (d_in[i] > max_val) max_val = d_in[i];
    }
    
	d_out[o] = max_val;
}



// Ajusta los valores de la primera capa
void input_adj(int LO_SZ, unsigned char* image, float* output, int mean, int std) {

	input_kernel<<<LO_SZ, LO_SZ>>>(output, image, LO_SZ, mean, std);
	
}


// Realiza una convolución.
void convolution(int LI, int LI_SZ, int LO, int LO_SZ, float *input, float *output, float* weight, float* bias) {

	dim3 blockSize = dim3(LI_SZ, LI_SZ);
	dim3 gridSize  = dim3(LO);
	
	if (LI == 1)
		conv_kernel<1, 32><<<gridSize, blockSize>>>(output, input, weight, bias, LO, LO_SZ);
	else if (LI == 6)
		conv_kernel<6, 14><<<gridSize, blockSize>>>(output, input, weight, bias, LO, LO_SZ);
	else
		conv_kernel<16, 5><<<gridSize, blockSize>>>(output, input, weight, bias, LO, LO_SZ);
}


// Realiza un submuestreo tomando el valor máximo
void subsamp_max(int LI, int LI_SZ, int LO, int LO_SZ, float* input, float* output) {
	
	int dim = LI_SZ / LO_SZ;
	
	dim3 blockSize(LO_SZ, LO_SZ, LO >> 1);
	int bx = (LO_SZ + blockSize.x - 1) / blockSize.x;
	int by = (LO_SZ + blockSize.y - 1) / blockSize.y;
	int bz = (LO + blockSize.z - 1) / blockSize.z;
	dim3 gridSize = dim3(bx, by, bz);
	
	subsamp_kernel<<<gridSize, blockSize>>>(output, input, dim, LI_SZ, LO, LO_SZ);

}

