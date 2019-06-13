#include "lenet.h"

// Funci�n relu
// if (x > 0) return x; else return 0;
decimal_t relu(decimal_t x) {
	return x*(x > 0);
}

void convolution_1(decimal_t input[INPUT_AMT * INPUT_SZ * INPUT_SZ], decimal_t output[LAYER1_AMT * LAYER1_SZ * LAYER1_SZ],
		decimal_t weight[INPUT_AMT * LAYER1_AMT * KERNEL_SZ * KERNEL_SZ], decimal_t bias[LAYER1_AMT]) {

	int i, o, w;

	for (int l2 = 0; l2 < LAYER1_AMT; l2++)
	for (int y = 0; y < LAYER1_SZ; y++)
	for (int x = 0; x < LAYER1_SZ; x++) {

			o = x + LAYER1_SZ * (y + l2 * LAYER1_SZ);

			for (int l1 = 0; l1 < INPUT_AMT; l1++)
			for (int k1 = 0; k1 < KERNEL_SZ; k1++)
			for (int k2 = 0; k2 < KERNEL_SZ; k2++) {

				i = x + k2 + INPUT_SZ * (y + k1 + INPUT_SZ * l1);
				w = k2 + KERNEL_SZ * (k1 + KERNEL_SZ * (l2 + l1 * LAYER1_AMT));
				output[o] += input[i] * weight[w];
			}

			decimal_t val = output[o] + bias[l2];
			output[o] = (val > 0.0) ? val : (decimal_t) 0.0;
	}


}


void convolution_2(decimal_t input[LAYER2_AMT * LAYER2_SZ * LAYER2_SZ], decimal_t output[LAYER3_AMT * LAYER3_SZ * LAYER3_SZ],
		decimal_t weight[LAYER2_AMT * LAYER3_AMT * KERNEL_SZ * KERNEL_SZ], decimal_t bias[LAYER3_AMT]) {

	int i, o, w;

	for (int l2 = 0; l2 < LAYER3_AMT; l2++)
	for (int y = 0; y < LAYER3_SZ; y++)
	for (int x = 0; x < LAYER3_SZ; x++) {

			o = x + LAYER3_SZ * (y + l2 * LAYER3_SZ);

			for (int l1 = 0; l1 < LAYER2_AMT; l1++)
			for (int k1 = 0; k1 < KERNEL_SZ; k1++)
			for (int k2 = 0; k2 < KERNEL_SZ; k2++) {
				i = x + k2 + LAYER2_SZ * (y + k1 + LAYER2_SZ * l1);
				w = k2 + KERNEL_SZ * (k1 + KERNEL_SZ * (l2 + l1 * LAYER3_AMT));

				output[o] += input[i] * weight[w];
			}

			decimal_t val = output[o] + bias[l2];
			output[o] = (val > 0.0) ? val : (decimal_t) 0.0;
	}
}


void convolution_3(decimal_t input[LAYER4_AMT * LAYER4_SZ * LAYER4_SZ], decimal_t output[LAYER5_AMT * LAYER5_SZ * LAYER5_SZ],
		decimal_t weight[LAYER4_AMT * LAYER5_AMT * KERNEL_SZ * KERNEL_SZ], decimal_t bias[LAYER5_AMT]) {

	int i, o, w;

	for (int l2 = 0; l2 < LAYER5_AMT; l2++)
	for (int y = 0; y < LAYER5_SZ; y++)
	for (int x = 0; x < LAYER5_SZ; x++) {

			o = x + LAYER5_SZ * (y + l2 * LAYER5_SZ);

			for (int l1 = 0; l1 < LAYER4_AMT; l1++)
			for (int k1 = 0; k1 < KERNEL_SZ; k1++)
			for (int k2 = 0; k2 < KERNEL_SZ; k2++) {
				i = x + k2 + LAYER4_SZ * (y + k1 + LAYER4_SZ * l1);
				w = k2 + KERNEL_SZ * (k1 + KERNEL_SZ * (l2 + l1 * LAYER5_AMT));

				output[o] += input[i] * weight[w];
			}

			decimal_t val = output[o] + bias[l2];
			output[o] = (val > 0.0) ? val : (decimal_t) 0.0;
	}
}


void subsamp_max_1(decimal_t input[LAYER1_AMT * LAYER1_SZ * LAYER1_SZ], decimal_t output[LAYER2_AMT * LAYER2_SZ * LAYER2_SZ]) {

	int i, ii, o;

	for (int l = 0; l < LAYER2_AMT; l++)
	for (int y = 0; y < LAYER2_SZ; y++)
	for (int x = 0; x < LAYER2_SZ; x++) {
		int k1 = 0, k2 = 0, ismax;

		o = x + LAYER2_SZ * (y + LAYER2_SZ * l);

		for (int l1 = 0; l1 < SUBSAMP_FACTOR; l1++)
		for (int l2 = 0; l2 < SUBSAMP_FACTOR; l2++) {

			i = x * SUBSAMP_FACTOR + k2 + LAYER1_SZ * (y * SUBSAMP_FACTOR + k1 + LAYER1_SZ * l);
			ii = x * SUBSAMP_FACTOR + l2 + LAYER1_SZ * (y * SUBSAMP_FACTOR + l1 + LAYER1_SZ * l);

			ismax = input[ii] > input[i];
			k1 += ismax * (l1 - k1);
			k2 += ismax * (l2 - k2);
		}

		i = x * SUBSAMP_FACTOR + k2 + LAYER1_SZ * (y * SUBSAMP_FACTOR + k1 + LAYER1_SZ * l);
		output[o] = input[i];
	}
}


void subsamp_max_2(decimal_t input[LAYER3_AMT * LAYER3_SZ * LAYER3_SZ], decimal_t output[LAYER4_AMT * LAYER4_SZ * LAYER4_SZ]) {

	int i, ii, o;

	for (int l = 0; l < LAYER4_AMT; l++)
	for (int y = 0; y < LAYER4_SZ; y++)
	for (int x = 0; x < LAYER4_SZ; x++) {
		int k1 = 0, k2 = 0, ismax;

		o = x + LAYER4_SZ * (y + LAYER4_SZ * l);

		for (int l1 = 0; l1 < SUBSAMP_FACTOR; l1++)
		for (int l2 = 0; l2 < SUBSAMP_FACTOR; l2++) {

			i = x * SUBSAMP_FACTOR + k2 + LAYER3_SZ * (y * SUBSAMP_FACTOR + k1 + LAYER3_SZ * l);
			ii = x * SUBSAMP_FACTOR + l2 + LAYER3_SZ * (y * SUBSAMP_FACTOR + l1 + LAYER3_SZ * l);

			ismax = input[ii] > input[i];
			k1 += ismax * (l1 - k1);
			k2 += ismax * (l2 - k2);
		}

		i = x * SUBSAMP_FACTOR + k2 + LAYER3_SZ * (y * SUBSAMP_FACTOR + k1 + LAYER3_SZ * l);
		output[o] = input[i];
	}
}


// Realiza un incremento para cada uno de los 10 d�gitos en funci�n de los valores
// de la capa anterior.
void dot_product(decimal_t input[LAYER5_AMT], decimal_t output[OUTPUT_AMT], decimal_t weight[LAYER5_AMT * OUTPUT_AMT], decimal_t bias[OUTPUT_AMT]) {

	for (int l = 0; l < LAYER5_AMT; l++)
		for (int d = 0; d < OUTPUT_AMT; d++)
			// Se pone "input[l][0][0]" porque se sabe que LI_SZ es 1.
			output[d] += input[l] * weight[l * OUTPUT_AMT + d];

	for (int d = 0; d < OUTPUT_AMT; d++)
		output[d] = relu(output[d] + bias[d]);

}


// Devuelve el d�gito con mayor porcentaje.
digit get_result(decimal_t* output)
{

	char result = 0;
	decimal_t max_percent = output[0];
	for (int o = 1; o < 10; o++) {
		decimal_t val = output[o];
		if (val > max_percent) {
			result = o;
			max_percent = val;
		}
	}
	return result;
}


// Realiza una propagaci�n hacia delante para ver cu�l es el d�gito.
digit predict(lenet5 *lenet, net *data, image img, int hw_mode, int allocate) {

	// Inicializamos a 0 la media y la desviaci�n t�pica.
	float mean = 0, std = 0;

	// Obtenemos los datos necesarios para calcular ambos valores:
	for (int y = 0; y < IMG_SZ; y++)
		for (int x = 0; x < IMG_SZ; x++) {
			mean += img[y][x];
			std += img[y][x] * img[y][x];
		}

	// Calculamos el valor:
	mean = mean / (IMG_SZ * IMG_SZ);

	std = sqrt(std / (IMG_SZ * IMG_SZ) - mean * mean);

	// La inicializaci�n de la capa INPUT se realiza en base
	// a estos dos par�metros calculados. NOTA: se utiliza
	// input[0] porque s�lo hay una capa de entrada.

	for (int y = 0; y < IMG_SZ; y++)
		for (int x = 0; x < IMG_SZ; x++)
			data->layers[0][INPUT_SZ * (y + PADDING) + x + PADDING] = (decimal_t)( (int)( ((int)img[y][x] - (int) mean) / (int) std));

	perf_counter c;

	c.start();
	if (hw_mode)
		convolution_1hw(data->layers[0],  data->layers[1], lenet->weights[0], lenet->bias[0], allocate);
	else
		convolution_1(data->layers[0],  data->layers[1], lenet->weights[0], lenet->bias[0]);
	c.stop();
	add_cycles(0, c.avg_cpu_cycles());

	c.start();
	subsamp_max_1(data->layers[1],  data->layers[2]);
	c.stop();
	add_cycles(1, c.avg_cpu_cycles());

	c.start();
	if (hw_mode)
		convolution_2hw(data->layers[2],  data->layers[3], lenet->weights[1], lenet->bias[1], allocate);
	else
		convolution_2(data->layers[2],  data->layers[3], lenet->weights[1], lenet->bias[1]);
	c.stop();
	add_cycles(2, c.avg_cpu_cycles());

	c.start();
	subsamp_max_2(data->layers[3],  data->layers[4]);
	c.stop();
	add_cycles(3, c.avg_cpu_cycles());

	c.start();
	if (hw_mode)
		convolution_3hw(data->layers[4],  data->layers[5], lenet->weights[2], lenet->bias[2], allocate);
	else
		convolution_3(data->layers[4],  data->layers[5], lenet->weights[2], lenet->bias[2]);
	c.stop();
	add_cycles(4, c.avg_cpu_cycles());

	c.start();
	dot_product(data->layers[5],  data->layers[6], lenet->weights[3], lenet->bias[3]);
	c.stop();
	add_cycles(5, c.avg_cpu_cycles());

	// Nos quedamos con el resultado que mayor porcentaje tenga:
	return get_result(data->layers[6]);
}


#define TIMERS 6

long long int times[TIMERS];
void reset_timers() { for (int i = 0; i < TIMERS; i++) times[i] = 0; }
void add_cycles(int i, long long int time) { times[i] += time; }
long long int get_cycles(int i) { return times[i]; }
double get_time(int i) { return ((1000.0 / sds_clock_frequency()) * (double) times[i]); };
