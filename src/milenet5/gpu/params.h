
///////////////////////////////////////
//           PARÁMETROS              //
///////////////////////////////////////

// Tamaño del filtro
#define KERNEL_SZ	5

// Dimensiones de cada capa
#define INPUT_SZ 	32
#define LAYER1_SZ	28
#define LAYER2_SZ	14
#define LAYER3_SZ	10
#define LAYER4_SZ	5
#define LAYER5_SZ	1
#define OUTPUT_SZ	10

// Cantidad de capas
#define INPUT_AMT	1
#define	LAYER1_AMT	6
#define	LAYER2_AMT	6
#define	LAYER3_AMT	16
#define	LAYER4_AMT	16
#define	LAYER5_AMT	120
#define	OUTPUT_AMT	10

// Parámetros
#define ALPHA		0.5
#define PADDING		2

#define IMG_SZ		28

// Definición de tipos:
typedef unsigned char pixel;
typedef unsigned char digit;
typedef pixel image[IMG_SZ][IMG_SZ];

// Parámetros de la red neuronal:
struct lenet5 {
	
	// Pesos para cada una de las capas
	float weight1[INPUT_AMT][LAYER1_AMT][KERNEL_SZ][KERNEL_SZ];
	float weight3[LAYER2_AMT][LAYER3_AMT][KERNEL_SZ][KERNEL_SZ];
	float weight5[LAYER4_AMT][LAYER5_AMT][KERNEL_SZ][KERNEL_SZ];
	float weight6[LAYER5_AMT][OUTPUT_AMT]; // Pesos en 'output'
	
	// Bias (valor de polarización) para cada capa
	float bias1[LAYER1_AMT];
	float bias3[LAYER3_AMT];
	float bias5[LAYER5_AMT];
	float bias6[OUTPUT_AMT]; // Bias en 'output'
	
	// NOTA: Las capas 2 y 4 son de muestreo, por eso no incorporan
	// bias ni pesos.
};
typedef struct lenet5 lenet5;


// Información auxiliar para las capas en GPU:
const int LAYER_SZ[] = {
	sizeof(float) * INPUT_AMT * IMG_SZ * IMG_SZ,
	sizeof(float) * INPUT_AMT * INPUT_SZ * INPUT_SZ,
	sizeof(float) * LAYER1_AMT * LAYER1_SZ * LAYER1_SZ,
	sizeof(float) * LAYER2_AMT * LAYER2_SZ * LAYER2_SZ,
	sizeof(float) * LAYER3_AMT * LAYER3_SZ * LAYER3_SZ,
	sizeof(float) * LAYER4_AMT * LAYER4_SZ * LAYER4_SZ,
	sizeof(float) * LAYER5_AMT * LAYER5_SZ * LAYER5_SZ
};

struct lenet5_gpu {
	// Información alojada en la GPU
	unsigned char* img_layer;
	float *d_w[3], *d_b[3], *d_l[6];
};
typedef struct lenet5_gpu lenet5_gpu;

// Información de la red neuronal (contiene los valores reales de la red):
struct net {
	
	// La CPU únicamente se encarga del cálculo de la última capa.
	float layer5[LAYER5_AMT][LAYER5_SZ][LAYER5_SZ];
	float output[OUTPUT_SZ];
	
};
typedef struct net net;


// Realiza una propagación hacia delante para ver cuál es el dígito.
digit predict(lenet5 *lenet, image img);



///////////////////////////////////////
// FUNCIONES IMPLEMENTADAS EN LA CPU //
///////////////////////////////////////

long long mseconds() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec*1000 + t.tv_usec/1000;
}


// Función relu
// if (x > 0) return x; else return 0;
float relu(float x) {
	return x*(x > 0);
}


// Realiza un incremento para cada uno de los 10 dígitos en función de los valores
// de la capa anterior.
void dot_product(int LI, int LI_SZ, int LO, int LO_SZ, float* input, float *output, float* weight, float* bias) {

	for (int l = 0; l < LI; l++)
		for (int d = 0; d < LO; d++)
			// Se pone "input[l][0][0]" porque se sabe que LI_SZ es 1.
			output[d] += input[l] * weight[l * LO + d];

	for (int d = 0; d < LO; d++)
		output[d] = relu(output[d] + bias[d]);
	
}
