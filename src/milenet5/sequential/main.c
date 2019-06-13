#include <stdlib.h>
#include <stdio.h>
#include <float.h>


#define FILE_TEST_IMAGE			"param/t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL			"param/t10k-labels-idx1-ubyte"
#define LENET_FILE 				"param/model.dat"
#define COUNT_TEST				10000

// #define DEBUG

#include "../lenet.h"
#include "../measure.h"


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


int testing(lenet5 *lenet, image *test_data, char *test_label,int total_size)
{
	
	int right = 0;
	for (int i = 0; i < total_size; ++i)
	{
		char l = test_label[i];
		int p = predict(lenet, test_data[i]);
		right += l == p;
		#ifdef DEBUG
		printf("Test %d  ➝  PRED [%d]  RES [%d]\n", i, p, l);
		#endif
	}
	return right;
}

int load(lenet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet, sizeof(lenet5), 1, fp);
	fclose(fp);
	return 0;
}


int main()
{
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	digit *test_label = (digit *)calloc(COUNT_TEST, sizeof(digit));
	
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("No se han encontrado los ficheros para las pruebas. Saliendo...");
		free(test_data);
		free(test_label);
		return 1;
	}

	lenet5 *lenet = (lenet5 *)malloc(sizeof(lenet5));
	if (load(lenet, LENET_FILE)) {
		printf("No se ha encontrado el fichero con los parámetros de la red neuronal. Saliendo...");
		free(lenet);
		free(test_data);
		free(test_label);
		return 2;
	}

	
	long long start = mseconds();
	int right = testing(lenet, test_data, test_label, COUNT_TEST);
	printf("Porcentaje de acierto: %d/%d\n", right, COUNT_TEST);
	printf("Tiempo: %lld ms\n", mseconds() - start);

	// Liberamos antes de salir
	free(lenet);
	free(test_data);
	free(test_label);
	
	return 0;
}

