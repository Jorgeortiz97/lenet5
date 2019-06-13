/* Fichero con los par�metros, definiciones y tipos de datos */
/* Tambi�n incluye funciones que son comunes a todas las versiones */
#ifndef _LENET_H
#define _LENET_H

#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "param.h"
#include "hw_kernels.h"
#include "sds_lib.h"


void reset_timers();
void add_cycles(int i, long long int time);
long long int get_cycles(int i);
double get_time(int i);


// Par�metros de la red neuronal:
struct lenet5
{
	// Pesos y valores de polarizaci�n para cada una de las capas
	decimal_t* weights[4];
	decimal_t* bias[4];
};
typedef struct lenet5 lenet5;


// Par�metros de la red neuronal:
struct lenet5_fixed
{
	// Pesos para cada una de las capas
	float weight1[INPUT_AMT  * LAYER1_AMT * KERNEL_SZ * KERNEL_SZ];
	float weight3[LAYER2_AMT * LAYER3_AMT * KERNEL_SZ * KERNEL_SZ];
	float weight5[LAYER4_AMT * LAYER5_AMT * KERNEL_SZ * KERNEL_SZ];
	float weight6[LAYER5_AMT * OUTPUT_AMT]; // Pesos en 'output'

	// Bias (valor de polarizaci�n) para cada capa
	float bias1[LAYER1_AMT];
	float bias3[LAYER3_AMT];
	float bias5[LAYER5_AMT];
	float bias6[OUTPUT_AMT]; // Bias en 'output'

};

typedef struct lenet5_fixed lenet5_fixed;

// Informaci�n de la red neuronal (contiene los valores reales de la red):
struct net
{
	decimal_t* layers[7];
};
typedef struct net net;

// Realiza una propagaci�n hacia delante para ver cu�l es el d�gito.
digit predict(lenet5 *lenet, net* data, image img, int hw_mode, int allocate);

#endif

