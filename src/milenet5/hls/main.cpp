#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <string.h>

#include "sds_lib.h"


const char FILE_TEST_IMAGE[] =			"param/t10k-images-idx3-ubyte";
const char FILE_TEST_LABEL[] =			"param/t10k-labels-idx1-ubyte";
const char LENET_FILE[] = 				"param/model.dat";

// #define DEBUG


#include "lenet.h"
#include "param.h"

int read_data(image* data, digit* label, const int count, const char data_file[], const char label_file[])
{
	FILE *fp_image = fopen(data_file, "rb");
	FILE *fp_label = fopen(label_file, "rb");
	if (!fp_image || !fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(image) * count, 1, fp_image);
	fread(label, count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}


int testing(lenet5 *lenet, net* data, image *test_data, digit *test_label, int total_size, int hw_mode) {

	int right = 0, p;
	for (int i = 0; i < total_size; ++i) {

		for (int l = 0; l < 7; l++)
			memset(data->layers[l], 0, LAYER_TOT_SZ[l]);

		char l = test_label[i];
		if (hw_mode)
			p = predict(lenet, data, test_data[i], 1, (i == 0));
		else
			p = predict(lenet, data, test_data[i], 0, 0);
		right += l == p;
		#ifdef DEBUG
		printf("Test %d  ->  PRED [%d]  RES [%d]\n", i, p, l);
		#endif
	}
	return right;
}

int load(lenet5 *lenet, lenet5_fixed *fixed, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(fixed, sizeof(lenet5_fixed), 1, fp);
	fclose(fp);

	for (int i = 0; i < INPUT_AMT   * LAYER1_AMT * KERNEL_SZ * KERNEL_SZ; i++) lenet->weights[0][i] = (decimal_t) fixed->weight1[i];
	for (int i = 0; i < LAYER2_AMT  * LAYER3_AMT * KERNEL_SZ * KERNEL_SZ; i++) lenet->weights[1][i] = (decimal_t) fixed->weight3[i];
	for (int i = 0; i < LAYER4_AMT  * LAYER5_AMT * KERNEL_SZ * KERNEL_SZ; i++) lenet->weights[2][i] = (decimal_t) fixed->weight5[i];

	for (int i = 0; i < LAYER5_AMT  * OUTPUT_AMT; i++) lenet->weights[3][i] = (decimal_t) fixed->weight6[i];
	for (int i = 0; i < LAYER1_AMT; i++) lenet->bias[0][i] = (decimal_t) fixed->bias1[i];
	for (int i = 0; i < LAYER3_AMT; i++) lenet->bias[1][i] = (decimal_t) fixed->bias3[i];
	for (int i = 0; i < LAYER5_AMT; i++) lenet->bias[2][i] = (decimal_t) fixed->bias5[i];
	for (int i = 0; i < OUTPUT_AMT; i++) lenet->bias[3][i] = (decimal_t) fixed->bias6[i];
	return 0;
}



void allocate_res(lenet5* lenet, net* data) {

	for (int i = 0; i < 4; i++) {
		lenet->weights[i] = (decimal_t *) sds_alloc(WEIGHT_TOT_SZ[i]);
		lenet->bias[i] = (decimal_t *) sds_alloc(BIAS_TOT_SZ[i]);
	}

	for (int i = 0; i < 7; i++)
		data->layers[i] = (decimal_t *) sds_alloc(LAYER_TOT_SZ[i]);
}


void free_res(lenet5* lenet, lenet5_fixed* fixed, net* data) {

	sds_free(fixed);

	for (int i = 0; i < 4; i++) {
		sds_free(lenet->weights[i]);
		sds_free(lenet->bias[i]);
	}
	sds_free(lenet);


	for (int i = 0; i < 7; i++)
		sds_free(data->layers[i]);
	sds_free(data);

}

int main(int argc, char* argv[])
{
	if (argc != 3) {
		printf("Uso ./Lenet5.elf [1, 10000] [SW_ENABLED]\n");
		return 1;
	}

	const int COUNT_TEST = atoi(argv[1]);
	const int SW_ENABLED = atoi(argv[2]);

	if (COUNT_TEST > 10000 || COUNT_TEST < 1) {
		printf("Valor incorrecto de casos: %d\n", COUNT_TEST);
		return 1;
	}

	image *test_data = (image *) sds_alloc(COUNT_TEST * sizeof(image));
	digit *test_label = (digit *) sds_alloc(COUNT_TEST * sizeof(digit));
	lenet5 *lenet = (lenet5 *) sds_alloc(sizeof(lenet5));
	lenet5_fixed *fixed = (lenet5_fixed *) sds_alloc(sizeof(lenet5_fixed));
	net *data = (net *) sds_alloc(sizeof(net));

	allocate_res(lenet, data);

	if (test_data == NULL || test_label == NULL || lenet == NULL || fixed == NULL || data == NULL)
		return 1;

	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("No se han encontrado los ficheros para las pruebas. Saliendo...");
		free_res(lenet, fixed, data);
		sds_free(test_data);
		sds_free(test_label);
		return 1;
	}

	if (load(lenet, fixed, (char *) LENET_FILE)) {
		printf("No se ha encontrado el fichero con los parametros de la red neuronal. Saliendo...");
		free_res(lenet, fixed, data);
		sds_free(test_data);
		sds_free(test_label);
		return 1;
	}

	printf("Iniciando pruebas...\n");

	perf_counter sw, hw;

	if (SW_ENABLED) {

		reset_timers();
		sw.start();
		int sw_rate = testing(lenet, data, test_data, test_label, COUNT_TEST, 0);
		sw.stop();

		printf("Implementado en SW:\n");
		printf("Cantidad de ciclos CPU: %lld (%f ms)\n", sw.avg_cpu_cycles(), sw.avg_cpu_ms());
		printf("Porcentaje de acierto: %d/%d\n", sw_rate, COUNT_TEST);
		/*printf("Ciclos operaciones:\n"
				"\t Conv1: %lld (%f ms)\n"
				"\t Conv2: %lld (%f ms)\n"
				"\t Conv3: %lld (%f ms)\n"
				"\t SubSam1: %lld (%f ms)\n"
				"\t SubSam2: %lld (%f ms)\n"
				"\t DotProd: %lld (%f ms)\n",
				get_cycles(0), get_time(0), get_cycles(1), get_time(1), get_cycles(2), get_time(2),
				get_cycles(3), get_time(3), get_cycles(4), get_time(4), get_cycles(5), get_time(5)
		);*/
	}

	reset_timers();
	hw.start();
	int hw_rate = testing(lenet, data, test_data, test_label, COUNT_TEST, 1);
	hw.stop();

	printf("\nImplementado en HW:\n");
	printf("Cantidad de ciclos CPU: %lld (%f ms)\n", hw.avg_cpu_cycles(), hw.avg_cpu_ms());
	/*printf("Ciclos operaciones:\n"
			"\t Conv1: %lld (%f ms)\n"
			"\t Conv2: %lld (%f ms)\n"
			"\t Conv3: %lld (%f ms)\n"
			"\t SubSam1: %lld (%f ms)\n"
			"\t SubSam2: %lld (%f ms)\n"
			"\t DotProd: %lld (%f ms)\n",
			get_cycles(0), get_time(0), get_cycles(1), get_time(1), get_cycles(2), get_time(2),
			get_cycles(3), get_time(3), get_cycles(4), get_time(4), get_cycles(5), get_time(5)
	);*/
	printf("Porcentaje de acierto: %d/%d\n", hw_rate, COUNT_TEST);

	free_res(lenet, fixed, data);
	sds_free(test_data);
	sds_free(test_label);

	return 0;
}

