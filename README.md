# Implementación y estudio de la red neuronal LeNet-5

Este repositorio contiene el código fuente utilizado en el Trabajo de Fin de Grado del Grado de Ingeniería Informática por la Universidad de Murcia.
* **Autor**: Jorge Ortiz Escribano.
* **Supervisor**: Juan Luis Aragón Alcaraz.

## Descripción
El presente proyecto consiste en el diseño, la implementación y posterior estudio de la red neuronal convolucional LeNet-5. Esta red neuronal se utiliza para el reconocimiento de dígitos. La estructura de la red neuronal es la siguiente:

![Estructura de Lenet-5](/doc/MiLenet5.png)

## Versiones implementadas

Las diferentes versiones que se han implementado son las siguientes:
* **Versión secuencial** . Diseñada para un ordenador de propósito general y escrita en C.
* **Versión OpenMP**. Diseñada para un ordenador de propósito general y escrita en C en combinación del framework OpenMP.
* **Versión GPU**. Diseñada combinando un ordenador de propósito general y unidad de procesamiento gráfico (GPU). Escrita en C usando el API de Nvidia CUDA.
* **Versión FPGA-HLS**. Sintetizada a partir de código de alto nivel (HLS). Escrita en C/C++ usando Vivado (SDSoC).
* **Versión FPGA-Verilog**. Escrita directamente usando lenguaje de descripción de hardware (HDL); en concreto, Verilog.

## Recursos empleados
En este apartado se comentan los recursos que se utilizarán para la elaboración del presente trabajo incluyendo tanto lo elementos software como las plataformas hardware.

#### Tecnologías software

* **gcc:** GNU C Compiler. Versión 6.5.0.
* **nvcc:** Nvidia C Compiler. Versión V9.1.85.
* **OpenMP:** Versión 11-2015.
* **GNU make:** Versión 4.1.
* **Putty:** Versión 0.7.
* **Git:** Versión 2.17.1.
* **Likwid:** Versión 4.3.3.
* **Vivado toolkit:** Versión 2018.3. Incluyendo Vivado, Vivado HLS y SDSoC.

#### Plataformas hardware
Las versiones implementadas para computadores de propósito general se han evaluado en una máquina compuesta por dos hexacores Intel Xeon E5-2667 v3. Ambos sockets componen una arquitectura NUMA a través de la memoria DRAM.

En concreto, la máquina presenta las siguientes características hardware:

* Dos procesadores Intel(R) Xeon E5-2667 v3. Lo que representa un total de 16 núcleos físicos, cada uno a 3.20 GHz, con 20 MB de caché e *hyperthreading*).
* 128 GB de memoria RAM.

Por otro lado, la versión acelerada por GPU se ha evaluado usando una tarjeta gráfica GeForce GTX 760, la cual presenta las siguientes características:

* Arquitectura Kepler.
* 192 CUDA *cores* (6 *Streaming Multiprocessors* o *SM* y 192 *Streaming Processors* por *SM*).
* 2 GB de memoria RAM GDDR5.

Con respecto al hardware reprogramable, se ha utilizado la placa ZedBoard, la cual incorpora el *System on Chip* Zynq-7000. Ese SoC combina un procesador Dual-core ARM Cortex™-A9 con una FPGA y 512 MB de memoria RAM DDR3. El procesador ARM funciona a una frecuencia de 667 MHz. Con respecto a la lógica reprogramable, la placa incorpora los siguientes recursos:

* 280 bloques RAM de 18 Kb (≈ 4.92 MB).
* 220 unidades DSP_48E.
* 106400 biestables (*FlipFlops*).
* 53200 LUTs (*Look-Up Table*).

## Resultados obtenidos

El porcentaje de éxito de la red es de 95.5% en el caso de las versiones que utilizan coma flotante de simple precisión. Para el caso de las versiones que utilizan coma fija como formato de los datos, el porcentaje de éxito se reduce a 95.45%.

En la siguiente gráfica se aprecia el rendimiento obtenido por cada plataforma, expresado como la cantidad de dígitos que se reconocen por segundo:

![Gráfica de rendimiento](/doc/Performance.jpg)

Si además del rendimiento, también se tiene en cuenta la potencia consumidad de cada plataforma, entonces se obtiene una métrica que permite conocer la cantidad de dígitos reconocidos por segundo por cada vatio consumido. A continuación, aparece una gráfica que refleja los resultados:

![Gráfica de rendimiento](/doc/PerformancePower.jpg)
