#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <sys/time.h>
#include <float.h>

const char FILE_TEST_IMAGE[] = "param/t10k-images-idx3-ubyte";
const char FILE_TEST_LABEL[] = "param/t10k-labels-idx1-ubyte";
const char LENET_FILE[] = "param/model.dat";
#define COUNT_TEST 10000


#include "params.h"
#include "kernels.cu"


// Devuelve el dígito con mayor porcentaje.
digit get_result(float* output) {
	
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
digit predict(lenet5 *lenet, lenet5_gpu *lenet_gpu, image img) {
	
	// Crea una red neuronal e inicializa a 0 todas sus capas.
	net nets = { 0 };

	// Inicializamos a 0 la media y la desviación típica.
	float mean = 0, std = 0;
	
	// Obtenemos los datos necesarios para calcular ambos valores:
	for (int y = 0; y < IMG_SZ; y++)
		for (int x = 0; x < IMG_SZ; x++) {
			mean += img[y][x];
			std += img[y][x] * img[y][x];
		}
	
	// Calculamos ambos valores:
	mean = mean / (IMG_SZ * IMG_SZ);
	std = sqrt(std / (IMG_SZ * IMG_SZ) - mean * mean);
	
	// Copiamos la imagen a la GPU:
	cudaMemcpy(lenet_gpu->img_layer, img, LAYER_SZ[0], cudaMemcpyHostToDevice);

	// Calculamos la capa 0:
	input_adj(INPUT_SZ, lenet_gpu->img_layer, lenet_gpu->d_l[0], mean, std);
	
	// Realizamos la propagación a partir de la capa 0:
	convolution(INPUT_AMT, INPUT_SZ, LAYER1_AMT, LAYER1_SZ,   lenet_gpu->d_l[0],  lenet_gpu->d_l[1], lenet_gpu->d_w[0], lenet_gpu->d_b[0]);
	subsamp_max(LAYER1_AMT, LAYER1_SZ, LAYER2_AMT, LAYER2_SZ, lenet_gpu->d_l[1],  lenet_gpu->d_l[2]);
	convolution(LAYER2_AMT, LAYER2_SZ, LAYER3_AMT, LAYER3_SZ, lenet_gpu->d_l[2],  lenet_gpu->d_l[3], lenet_gpu->d_w[1], lenet_gpu->d_b[1]);
	subsamp_max(LAYER3_AMT, LAYER3_SZ, LAYER4_AMT, LAYER4_SZ, lenet_gpu->d_l[3],  lenet_gpu->d_l[4]);
	convolution(LAYER4_AMT, LAYER4_SZ, LAYER5_AMT, LAYER5_SZ, lenet_gpu->d_l[4],  lenet_gpu->d_l[5], lenet_gpu->d_w[2], lenet_gpu->d_b[2]);
	
	// Copiamos el resultado a la CPU:
	cudaMemcpy(nets.layer5, lenet_gpu->d_l[5], LAYER_SZ[6], cudaMemcpyDeviceToHost);
	
	// Llevamos a cabo el cálculo final:
	dot_product(LAYER5_AMT, LAYER5_SZ, OUTPUT_AMT, OUTPUT_SZ, (float *) nets.layer5, (float *) nets.output, (float *) lenet->weight6, (float *) lenet->bias6);
		
	// Nos quedamos con el resultado que mayor porcentaje tenga:
	return get_result(nets.output);
}



int read_data(image* data, digit* label, const int count, const char data_file[], const char label_file[]) {
	
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


int testing(lenet5 *lenet, lenet5_gpu *lenet_gpu, image *test_data, digit *test_label,int total_size) {
	
	int right = 0;
	for (int i = 0; i < total_size; ++i) {
		char l = test_label[i];
		int p = predict(lenet, lenet_gpu, test_data[i]);
		right += l == p;
		#ifdef DEBUG
		printf("Test %d  ➝  PRED [%d]  RES [%d]\n", i, p, l);
		#endif
	}
	return right;
}


int load(lenet5 *lenet, char filename[]) {
	// Cargamos los parámetros en la memoria de la CPU:
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet, sizeof(lenet5), 1, fp);
	fclose(fp);
	return 0;
}

const int weight_sz[] = {
	sizeof(float) * INPUT_AMT * LAYER1_AMT * KERNEL_SZ * KERNEL_SZ,
	sizeof(float) * LAYER2_AMT * LAYER3_AMT * KERNEL_SZ * KERNEL_SZ,
	sizeof(float) * LAYER4_AMT * LAYER5_AMT * KERNEL_SZ * KERNEL_SZ,	
};
const int bias_sz[] = {
	sizeof(float) * LAYER1_AMT,
	sizeof(float) * LAYER3_AMT,
	sizeof(float) * LAYER5_AMT,
};
	
void load_gpu(lenet5 *lenet, lenet5_gpu *lenet_gpu) {

	// Reservamos memoria en la GPU y copiamos los parámetros:
	for (int i = 0; i < 3; i++) {
		cudaMalloc((void **) &lenet_gpu->d_w[i], weight_sz[i]);
		cudaMalloc((void **) &lenet_gpu->d_b[i], bias_sz[i]);
	}
	
	cudaMemcpy(lenet_gpu->d_w[0], lenet->weight1, weight_sz[0], cudaMemcpyHostToDevice); 
	cudaMemcpy(lenet_gpu->d_w[1], lenet->weight3, weight_sz[1], cudaMemcpyHostToDevice); 
	cudaMemcpy(lenet_gpu->d_w[2], lenet->weight5, weight_sz[2], cudaMemcpyHostToDevice); 	
	cudaMemcpy(lenet_gpu->d_b[0], lenet->bias1, bias_sz[0], cudaMemcpyHostToDevice); 
	cudaMemcpy(lenet_gpu->d_b[1], lenet->bias3, bias_sz[1], cudaMemcpyHostToDevice); 
	cudaMemcpy(lenet_gpu->d_b[2], lenet->bias5, bias_sz[2], cudaMemcpyHostToDevice); 	
	
	cudaMalloc((void **) &lenet_gpu->img_layer, LAYER_SZ[0]);
	
	for (int i = 0; i < 6; i++)
		cudaMalloc((void **) &lenet_gpu->d_l[i], LAYER_SZ[i+1]);
}

int main(int argc, char* argv[]) {
	
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	digit *test_label = (digit *)calloc(COUNT_TEST, sizeof(digit));
	
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL)) {
		printf("No se han encontrado los ficheros para las pruebas. Saliendo...");
		free(test_data);
		free(test_label);
		return 1;
	}

	lenet5 *lenet;
	lenet5_gpu *lenet_gpu;
	if (cudaMallocHost((void**) &lenet, sizeof(lenet5)) != cudaSuccess ||
		cudaMallocHost((void**) &lenet_gpu, sizeof(lenet_gpu)) != cudaSuccess) {
		
		free(test_data);
		free(test_label);
		return 1;	
	}
			
	if (load(lenet, (char *) LENET_FILE)) {
		printf("No se ha encontrado el fichero con los parámetros de la red neuronal. Saliendo...");
		cudaFreeHost(lenet);
		cudaFreeHost(lenet_gpu);
		free(test_data);
		free(test_label);
		return 1;
	}
	
	// Inicio del contador:
	long long start = mseconds();
	
	// Carga de datos en la GPU:
	load_gpu(lenet, lenet_gpu);

	// Inicio del procesamiento en la GPU:
	int right = testing(lenet, lenet_gpu, test_data, test_label, COUNT_TEST);
	
	// Liberado de memoria en la GPU:
	for (int i = 0; i < 3; i++) {
		cudaFree(lenet_gpu->d_w[i]);
		cudaFree(lenet_gpu->d_b[i]);
	}
	for (int i = 0; i < 6; i++)
		cudaFree(lenet_gpu->d_l[i]);
	
	// Fin del contador:
	long long end = mseconds();
	
	
	// Mostrado del tiempo:
	printf("Porcentaje de acierto: %d/%d\n", right, COUNT_TEST);
	printf("Tiempo: %lld ms\n", end - start);

	
	// Liberamos la memoria en la CPU antes de salir:
	cudaFreeHost(lenet);
	cudaFreeHost(lenet_gpu);
	free(test_data);
	free(test_label);
	
	return 0;
}


