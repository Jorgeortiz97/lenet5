#include "../lenet.h"

// Función relu
// if (x > 0) return x; else return 0;
float relu(float x)
{
	return x*(x > 0);
}



/* TERMINOLOGÍA A UTILIZAR EN LAS DIFERENTES CAPAS
 + LI: cantidad de capas de entrada.
 + LI_SZ: dimensión de cada capa de entrada (tamaño: LI_SZ x LI_SZ).
 + LO: cantidad de capas de salida.
 + LO_SZ: dimensión de cada capa de salida (tamaño: LO_SZ x LO_SZ).

 + Nombrado de los índices:
	- l1: índice para la capa actual de salida.
	- l2: índice para la capa actual de entrada.
	- y: índice para la primera dimensión de la capa actual de salida.
	- x: índice para la segunda dimensión de la capa actual de salida.
	- k1: índice para la primera dimensión del filtro (kernel).
	- k2: índice para la segunda dimensión del filtro (kernel).
	- d: índice para el dígito sobre el que se está trabajando.
*/


// Realiza una convolución.
void convolution(int LI, int LI_SZ, int LO, int LO_SZ, float input[LI][LI_SZ][LI_SZ], float output[LO][LO_SZ][LO_SZ], float weight[LI][LO][KERNEL_SZ][KERNEL_SZ], float bias[LO_SZ]) {

	#pragma omp parallel for collapse (3)
	for (int l2 = 0; l2 < LO; l2++)
	for (int y = 0; y < LO_SZ; y++)
	for (int x = 0; x < LO_SZ; x++) {
		for (int l1 = 0; l1 < LI; l1++)
		for (int k1 = 0; k1 < KERNEL_SZ; k1++)
		for (int k2 = 0; k2 < KERNEL_SZ; k2++)
			output[l2][y][x] += input[l1][y + k1][x + k2] * weight[l1][l2][k1][k2];
		output[l2][y][x] = relu(output[l2][y][x] + bias[l2]);
	}
}

// Realiza un submuestreo tomando el valor máximo
void subsamp_max(int LI, int LI_SZ, int LO, int LO_SZ, float input[LI][LI_SZ][LI_SZ], float output[LO][LO_SZ][LO_SZ]) {
	int dim = LI_SZ / LO_SZ;

	#pragma omp parallel for collapse(3)
	for (int l = 0; l < LO; l++)
	for (int y = 0; y < LO_SZ; y++)
	for (int x = 0; x < LO_SZ; x++) {
		float max_val = -FLT_MAX;
		for (int l1 = 0; l1 < dim; l1++)
		for (int l2 = 0; l2 < dim; l2++)
			if (input[l][y * dim + l1][x * dim + l2] > max_val)
			    max_val = input[l][y * dim + l1][x * dim + l2];
		output[l][y][x] = max_val;
	}
}

// Realiza un incremento para cada uno de los 10 dígitos en función de los valores
// de la capa anterior.
void dot_product(int LI, int LI_SZ, int LO, int LO_SZ, float input[LI][LI_SZ][LI_SZ], float output[LO_SZ], float weight[LI][LO], float bias[LO]) {

	for (int l = 0; l < LI; l++)
		for (int d = 0; d < LO; d++)
			// Se pone "input[l][0][0]" porque se sabe que LI_SZ es 1.
			output[d] += input[l][0][0] * weight[l][d];
	#pragma simd
	for (int d = 0; d < LO; d++)
		output[d] = relu(output[d] + bias[d]);
	
}


// Devuelve el dígito con mayor porcentaje.
digit get_result(float* output)
{
	
	char result = 0;
	float max_percent = output[0];
	for (int o = 1; o < 10; o++) {
		float val = output[o];
		if (val > max_percent) {
			result = o;
			max_percent = val;
		}
	}
	return result;
}

// Realiza una propagación hacia delante para ver cuál es el dígito.
digit predict(lenet5 *lenet, image img) {
	
	// Crea una red neuronal e inicializa a 0 todas sus capas.
	net nets = { 0 };

	// Inicializamos a 0 la media y la desviación típica.
	float mean = 0, std = 0;
	
	// Obtenemos los datos necesarios para calcular ambos valores:
	#pragma simd
	for (int y = 0; y < IMG_SZ; y++)
		for (int x = 0; x < IMG_SZ; x++) {
			mean += img[y][x];
			std += img[y][x] * img[y][x];
		}
	
	// Calculamos el valor:
	mean = mean / (IMG_SZ * IMG_SZ);

	std = sqrt(std / (IMG_SZ * IMG_SZ) - mean * mean);
	
	
	// La inicialización de la capa INPUT se realiza en base
	// a estos dos parámetros calculados. NOTA: se utiliza
	// input[0] porque sólo hay una capa de entrada.
	#pragma simd
	for (int y = 0; y < IMG_SZ; y++)
		for (int x = 0; x < IMG_SZ; x++)
			nets.input[0][y + PADDING][x + PADDING] = (float)( (int)( ((int)img[y][x] - (int) mean) / (int) std));

	// Realizamos la propagación a partir de la capa INPUT:
	convolution(INPUT_AMT, INPUT_SZ, LAYER1_AMT, LAYER1_SZ, nets.input, nets.layer1, lenet->weight1, lenet->bias1);
	subsamp_max(LAYER1_AMT, LAYER1_SZ, LAYER2_AMT, LAYER2_SZ, nets.layer1, nets.layer2);
	convolution(LAYER2_AMT, LAYER2_SZ, LAYER3_AMT, LAYER3_SZ, nets.layer2, nets.layer3, lenet->weight3, lenet->bias3);
	subsamp_max(LAYER3_AMT, LAYER3_SZ, LAYER4_AMT, LAYER4_SZ, nets.layer3, nets.layer4);
	convolution(LAYER4_AMT, LAYER4_SZ, LAYER5_AMT, LAYER5_SZ, nets.layer4, nets.layer5, lenet->weight5, lenet->bias5);
	dot_product(LAYER5_AMT, LAYER5_SZ, OUTPUT_AMT, OUTPUT_SZ, nets.layer5, nets.output, lenet->weight6, lenet->bias6);
	
	
	// Nos quedamos con el resultado que mayor porcentaje tenga:
	return get_result(nets.output);
}
