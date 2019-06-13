#include <stdio.h>
#include "xtime_l.h"

#include "platform.h"
#include "xbasic_types.h"
#include "xparameters.h"

Xuint32 *baseaddr_p = (Xuint32 *)XPAR_CONVOLUTION_0_S00_AXI_BASEADDR;


// Acceso a la memoria SRAM (a través del bus AXI)
#define READ(addrOffset) *(baseaddr_p+addrOffset)
#define WRITE(addrOffset, value) *(baseaddr_p+addrOffset)=value


// Puertos de la memoria SRAM
#define BANK_ADDR		0x00

#define RESET			0x1A
#define STEP			0x1B

#define BANK_X			0x1E
#define BANK_Y			0x1F


#define BANK_ILAYER		0x01
#define BANK_LAYER1		0x02
#define BANK_LAYER2		0x03
#define BANK_LAYER3		0x04
#define BANK_LAYER4		0x05
#define BANK_LAYER5		0x06
#define BANK_OLAYER		0x07

#define BANK_WEIGHT1	0x11
#define BANK_WEIGHT3	0x12
#define BANK_WEIGHT5	0x13
#define BANK_WEIGHT6	0x14

#define BANK_BIAS1		0x15
#define BANK_BIAS3		0x16
#define BANK_BIAS5		0x17
#define BANK_BIAS6		0x18

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


void init_sram(unsigned int LI, unsigned int LO, unsigned int Y, unsigned int X, unsigned int BANK, unsigned int value) {

	for (int li = 0; li < LI; li++)
		for (int lo = 0; lo < LO; lo++)
			for (int y = 0; y < Y; y++)
				for (int x = 0; x < X; x++) {
					unsigned int addr = x + X * (y + Y * (lo + LO * li));
					WRITE(BANK_ADDR, addr);
					WRITE(BANK, value);
				}
}

void init_sram_variable(unsigned int LI, unsigned int LO, unsigned int Y, unsigned int X, unsigned int BANK, unsigned int value) {

	for (int li = 0; li < LI; li++)
		for (int lo = 0; lo < LO; lo++)
			for (int y = 0; y < Y; y++)
				for (int x = 0; x < X; x++) {
					unsigned int addr = x + X * (y + Y * (lo + LO * li));
					WRITE(BANK_ADDR, addr);
					WRITE(BANK, value+x);
				}
}

unsigned int check_sram(unsigned int LI, unsigned int LO, unsigned int Y, unsigned int X, unsigned int BANK, unsigned int value) {

	unsigned int miss = 0;

	for (int li = 0; li < LI; li++)
		for (int lo = 0; lo < LO; lo++)
			for (int y = 0; y < Y; y++)
				for (int x = 0; x < X; x++) {
					unsigned int addr = x + X * (y + Y * (lo + LO * li));
					WRITE(BANK_ADDR, 0);
					unsigned int val = READ(BANK);
					if (val != value)
						miss++;
				}
	return miss;
}



int main() {

	init_platform();

	xil_printf("Lenet-5 implementation for Verilog!\n\r");

	WRITE(RESET, 1);

	// Inicialización de la capa de entrada
	init_sram(1, 1, INPUT_SZ, INPUT_SZ, BANK_ILAYER, 0x00001000);

	// Inicialización de los pesos
	init_sram(1, LAYER1_AMT, KERNEL_SZ, KERNEL_SZ, BANK_WEIGHT1, 0x00000400);
	init_sram(LAYER2_AMT, LAYER3_AMT, KERNEL_SZ, KERNEL_SZ, BANK_WEIGHT3, 0x00000400);
	init_sram(LAYER4_AMT, LAYER5_AMT, KERNEL_SZ, KERNEL_SZ, BANK_WEIGHT5, 0x00000400);
	init_sram(LAYER5_AMT, OUTPUT_AMT, 1, 1, BANK_WEIGHT6, 0x00010000);

	// Inicialización de los valores de polarización
	init_sram(1, LAYER1_AMT, 1, 1, BANK_BIAS1, 0x00000000);
	init_sram(1, LAYER3_AMT, 1, 1, BANK_BIAS3, 0x00000000);
	init_sram(1, LAYER5_AMT, 1, 1, BANK_BIAS5, 0x00000000);
	init_sram(1, OUTPUT_AMT, 1, 1, BANK_BIAS6, 0x00000000);

	XTime tStart, tEnd;

	XTime_GetTime(&tStart);

	// Inicia el cómputo
	WRITE(RESET, 0);

	// Espera a que el resultado esté listo
	while (READ(STEP) != 6);

	XTime_GetTime(&tEnd);


	xil_printf("Coefficients for the output layer\n\r");
	for (unsigned int v = 0; v < 10; v++) {
		WRITE(BANK_ADDR, v);
		xil_printf("%u: %X\n\r", v, READ(BANK_OLAYER));
	}

	long long unsigned int cycles = (tEnd - tStart) * 2;
	long double elapsedTime = (long double)cycles / (long double)XPAR_PS7_CORTEXA9_0_CPU_CLK_FREQ_HZ;

    printf("Processing took %llu clock cycles.\n", cycles);
    printf("Elapsed time: %.5lf ms.\n", elapsedTime * 1000) ;


	return 0;
}
