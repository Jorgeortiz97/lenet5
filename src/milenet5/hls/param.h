#ifndef PARAM_H
#define PARAM_H

#include "ap_fixed.h"
#include "sds_lib.h"


// Tamaño del filtro
#define KERNEL_SZ	5
#define SUBSAMP_FACTOR	2

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

// Tipo de datos a usar (descomentar el que se quiera usar):
typedef ap_fixed<16,8> decimal_t;
//typedef float decimal_t;


const unsigned int LAYER_TOT_SZ[] = {
	sizeof(decimal_t) * INPUT_AMT   * INPUT_SZ  * INPUT_SZ,
	sizeof(decimal_t) * LAYER1_AMT   * LAYER1_SZ  * LAYER1_SZ,
	sizeof(decimal_t) * LAYER2_AMT   * LAYER2_SZ  * LAYER2_SZ,
	sizeof(decimal_t) * LAYER3_AMT   * LAYER3_SZ  * LAYER3_SZ,
	sizeof(decimal_t) * LAYER4_AMT   * LAYER4_SZ  * LAYER4_SZ,
	sizeof(decimal_t) * LAYER5_AMT   * LAYER5_SZ  * LAYER5_SZ,
	sizeof(decimal_t) * OUTPUT_SZ
};

const unsigned int WEIGHT_TOT_SZ[] = {
	sizeof(decimal_t) * INPUT_AMT  * LAYER1_AMT * KERNEL_SZ * KERNEL_SZ,
	sizeof(decimal_t) * LAYER2_AMT * LAYER3_AMT * KERNEL_SZ * KERNEL_SZ,
	sizeof(decimal_t) * LAYER4_AMT * LAYER5_AMT * KERNEL_SZ * KERNEL_SZ,
	sizeof(decimal_t) * LAYER5_AMT * OUTPUT_AMT
};

const unsigned int BIAS_TOT_SZ[] = {
	sizeof(decimal_t) * LAYER1_AMT,
	sizeof(decimal_t) * LAYER3_AMT,
	sizeof(decimal_t) * LAYER5_AMT,
	sizeof(decimal_t) * OUTPUT_AMT
};


class perf_counter
{
public:
     long long int tot, cnt, calls;
     perf_counter() : tot(0), cnt(0), calls(0) {};
     inline void reset() { tot = cnt = calls = 0; }
     inline void start() { cnt = sds_clock_counter(); calls++; };
     inline void stop() { tot += (sds_clock_counter() - cnt); };
     inline long long int avg_cpu_cycles() { return ((tot+(calls>>1)) / calls); };
     inline double avg_cpu_ms() { return ((1000.0 / sds_clock_frequency()) * (double) avg_cpu_cycles()); };
};


#endif
