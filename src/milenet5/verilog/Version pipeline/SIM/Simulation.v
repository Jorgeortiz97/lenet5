
`timescale 1 ns / 1 ps

module Lenet #
(
    parameter integer DIM           = 2,
    parameter integer KERNEL_SZ     = 5,   
    
    parameter integer INPUT_SZ      = 32,
    parameter integer LAYER1_SZ     = 28,
    parameter integer LAYER2_SZ     = 14,
    parameter integer LAYER3_SZ     = 10,
    parameter integer LAYER4_SZ     = 5,
    parameter integer LAYER5_SZ     = 1,
    
    parameter integer INPUT_AMT     = 1,
    parameter integer LAYER1_AMT    = 6,
    parameter integer LAYER2_AMT    = 6,
    parameter integer LAYER3_AMT    = 16,
    parameter integer LAYER4_AMT    = 16,
    parameter integer LAYER5_AMT    = 120,
    parameter integer OUTPUT_AMT    = 10,

    // Valores para el estado de la variable 'stage'
    parameter integer ADDRESSING    = 0,
    parameter integer FETCHING      = 1, 
    parameter integer EXECUTING     = 2,
    parameter integer ADJUSTING     = 3, 
    
    // Auto generated code
    parameter integer C_S_AXI_DATA_WIDTH	= 32,
    parameter integer C_S_AXI_ADDR_WIDTH	= 11
)();

reg clk = 0, reset = 0;

////////////////////////////////////////////
//       REGISTROS DE CONTROL             //
////////////////////////////////////////////

reg [C_S_AXI_DATA_WIDTH-1:0]	addr = 0; 

////////////////////////////////////////////
//       CAPAS DE LA RED NEURONAL         //
////////////////////////////////////////////

// Direcciones de acceso y valores de entrada
reg [C_S_AXI_DATA_WIDTH-1:0] il_addr, il_din, l1_addr, l1_din, l2_addr, l2_din,
                            l3_addr, l3_din, l4_addr, l4_din, l5_addr, l5_din,
                            ol_addr, ol_din;

// Valores de salida
wire signed [C_S_AXI_DATA_WIDTH-1:0] il_dout, l1_dout, l2_dout, l3_dout, l4_dout, l5_dout, ol_dout;

// Señal de escritura
reg il_en, l1_en, l2_en, l3_en, l4_en, l5_en, ol_en;    

// Capas (memoria SRAM)
SRAM #(10, 1024)    input_layer(clk, il_addr, il_en, il_din, il_dout);
SRAM #(13, 4704)    layer1(clk, l1_addr, l1_en, l1_din, l1_dout);
SRAM #(11, 1176)    layer2(clk, l2_addr, l2_en, l2_din, l2_dout);
SRAM #(11, 1600)    layer3(clk, l3_addr, l3_en, l3_din, l3_dout);
SRAM #(9, 400)      layer4(clk, l4_addr, l4_en, l4_din, l4_dout);
SRAM #(7, 120)      layer5(clk, l5_addr, l5_en, l5_din, l5_dout);
SRAM #(4, 10)       output_layer(clk, ol_addr, ol_en, ol_din, ol_dout);


////////////////////////////////////////////
//       PESOS DE LA RED NEURONAL         //
////////////////////////////////////////////

// Direcciones de acceso y valores de entrada
reg [C_S_AXI_DATA_WIDTH-1:0] w1_addr, w1_din, w3_addr, w3_din, w5_addr, w5_din, w6_addr, w6_din;

// Valores de salida
wire signed [C_S_AXI_DATA_WIDTH-1:0] w1_dout, w3_dout, w5_dout, w6_dout;

// Señal de escritura
reg w1_en, w3_en, w5_en, w6_en;    

// Pesos (memoria SRAM)
SRAM #(8, 150)          weight1(clk, w1_addr, w1_en, w1_din, w1_dout);
SRAM #(12, 2400)        weight3(clk, w3_addr, w3_en, w3_din, w3_dout);
SRAM #(16, 48000)       weight5(clk, w5_addr, w5_en, w5_din, w5_dout);
SRAM #(11, 1200)        weight6(clk, w6_addr, w6_en, w6_din, w6_dout);

////////////////////////////////////////////////
// VALORES DE POLARIZACIÓN DE LA RED NEURONAL //
////////////////////////////////////////////////

// Direcciones de acceso y valores de entrada
reg [C_S_AXI_DATA_WIDTH-1:0] b1_addr, b1_din, b3_addr, b3_din, b5_addr, b5_din, b6_addr, b6_din;

// Valores de salida
wire signed [C_S_AXI_DATA_WIDTH-1:0] b1_dout, b3_dout, b5_dout, b6_dout;

// Señal de escritura
reg b1_en, b3_en, b5_en, b6_en;

// Valores de polarización (memoria SRAM)
SRAM #(3, 6)      bias1(clk, b1_addr, b1_en, b1_din, b1_dout);
SRAM #(4, 16)     bias3(clk, b3_addr, b3_en, b3_din, b3_dout);
SRAM #(7, 120)    bias5(clk, b5_addr, b5_en, b5_din, b5_dout);
SRAM #(4, 10)     bias6(clk, b6_addr, b6_en, b6_din, b6_dout);


integer limit_xy[] = {LAYER1_SZ, LAYER2_SZ, LAYER3_SZ, LAYER4_SZ, LAYER5_SZ, LAYER5_SZ, 1};
integer limit_l1[] = {INPUT_AMT, 1, LAYER2_AMT, 1, LAYER4_AMT, LAYER4_AMT, LAYER5_AMT};
integer limit_l2[] = {LAYER1_AMT, LAYER2_AMT, LAYER3_AMT, LAYER4_AMT, LAYER5_AMT/2, LAYER5_AMT/2, OUTPUT_AMT};
integer limit_k[]  = {KERNEL_SZ, DIM, KERNEL_SZ, DIM, KERNEL_SZ, KERNEL_SZ, 1};


//////////////////////////////////////////
///////////// SIMULACIÓN /////////////////
//////////////////////////////////////////

always #5 clk = ~clk;

initial begin
    reset = 1;
    #200 reset = 0;
end

reg [31:0] step = 0;
reg [15:0] l1, l2, x, y, k1, k2;
reg [1:0] stage = 0;

reg signed [63:0] mul;
reg signed [31:0] aux = 0;



// Límites
reg [31:0] LIMIT_K, LIMIT_L1, LIMIT_XY, LIMIT_L2;


always @(posedge clk, posedge reset)
begin

    // Señal de reset
    if (reset) begin
    
        step        <= 0;
        l1          <= 0;
        l2          <= 0;
        x           <= 0;
        y           <= 0;
        k1          <= 0;
        k2          <= 0; 
        mul         <= 0;              
        aux         <= 0;
        stage       <= 0;
        LIMIT_K     <= KERNEL_SZ;
        LIMIT_XY    <= LAYER1_SZ;        
        LIMIT_L1    <= INPUT_AMT;
        LIMIT_L2    <= LAYER1_AMT;
        l1_en       <= 1;
        l2_en       <= 0;
        l3_en       <= 0;
        l4_en       <= 0;
        l5_en       <= 0;
        ol_en       <= 0;
        
    end else begin
    
        if (step < 6) begin
            // Control de los pasos
            if (l2 == LIMIT_L2) begin
            
                stage       <= 0;
                k2          <= 0;
                k1          <= 0;
                l1          <= 0;
                x           <= 0;
                y           <= 0;
                l2          <= 0;
                aux         <= 0;
                
                // Avanzamos de paso y recalculamos límites
                step <= step + 1;

                if (step == 0 || step == 2)
                    LIMIT_K     <= DIM;
                else if (step == 1 || step == 3)
                    LIMIT_K     <= KERNEL_SZ;
                else
                    LIMIT_K     <= 1;
                 
                if (step == 0) begin
                    LIMIT_XY    <=  LAYER2_SZ;
                    LIMIT_L1    <=  1;
                    LIMIT_L2    <=  LAYER2_AMT;
                end else if (step == 1) begin
                    LIMIT_XY    <=  LAYER3_SZ;
                    LIMIT_L1    <=  LAYER2_AMT;
                    LIMIT_L2    <=  LAYER3_AMT;
                end else if (step == 2) begin
                    LIMIT_XY    <=  LAYER4_SZ;
                    LIMIT_L1    <=  1;
                    LIMIT_L2    <=  LAYER4_AMT;
                end else if (step == 3) begin
                    LIMIT_XY    <=  LAYER5_SZ;
                    LIMIT_L1    <=  LAYER4_AMT;
                    LIMIT_L2    <=  LAYER5_AMT ;
                end else if (step == 4) begin
                    LIMIT_XY    <=  1;
                    LIMIT_L1    <=  LAYER5_AMT;
                    LIMIT_L2    <=  OUTPUT_AMT;
                end else if (step == 5) begin
                    $stop;
                end
                
                // Establecemos el modo escritura para la capa correspondiente de salida:
                l1_en       <= 0;
                l2_en       <= (step == 0);
                l3_en       <= (step == 1);
                l4_en       <= (step == 2);
                l5_en       <= (step == 3);
                ol_en       <= (step == 4);
                
                
            end else begin
       
				// FASE DE AJUSTE DE ÍNDICES:
                if (k2 == LIMIT_K) begin
                    k2  <= 0;
                    if (k1 == LIMIT_K - 1) begin
                        k1 <= 0;
                        
                        if (step == 1 || step == 3) begin
                        
                            if (step == 1)
                                l2_din  <= aux;
                            else if (step == 3)
                                l4_din  <= aux;
                        end
                        
                        if (l1 == LIMIT_L1 - 1) begin
                            l1 <= 0;
                            
                            if (step != 1 && step != 3) begin
                                
                                if (step == 0) begin
                                    if (aux + b1_dout > 0)
                                        l1_din  <= aux + b1_dout;
                                    else
                                        l1_din  <= 0;
                                end else if (step == 2) begin
                                    if (aux + b3_dout > 0)
                                        l3_din  <= aux + b3_dout;
                                    else
                                        l3_din  <= 0;
                                end else if (step == 4) begin
                                    if (aux + b5_dout > 0)
                                        l5_din  <= aux + b5_dout;
                                    else
                                        l5_din  <= 0;
                                end else if (step == 5) begin
                                    if (aux + b6_dout > 0)
                                        ol_din  <= aux + b6_dout;
                                    else
                                        ol_din  <= 0;
                                end
                                aux <= 0;
                            end
                            
                            if (x == LIMIT_XY - 1) begin
                                x <= 0;
                                if (y == LIMIT_XY - 1) begin
                                    y  <= 0;
                                    l2 <= l2 + 1;
                                end else
                                    y  <= y + 1;
                            end else
                                x <= x + 1;
                        end else
                            l1 <= l1 + 1;
                    end else
                        k1 <= k1 + 1;
                
                
                // FASE DE DIRECCIONAMIENTO
                end else if (stage == ADDRESSING) begin

                    // Direcciones de convolución:
                    if (step == 0) begin
                        il_addr     <= k2 + x + INPUT_SZ * (y + k1);
                        l1_addr     <= x + LAYER1_SZ * (y + LAYER1_SZ * l2);
                        b1_addr     <= l2;
                        w1_addr     <= k2 + KERNEL_SZ * (k1 + KERNEL_SZ * (l2 + LAYER1_AMT * l1));
                    end else if (step == 2) begin
                        l2_addr     <= k2 + x + LAYER2_SZ * (y + k1 + LAYER2_SZ * l1);
                        l3_addr     <= x + LAYER3_SZ * (y + LAYER3_SZ * l2);
                        b3_addr     <= l2;
                        w3_addr     <= k2 + KERNEL_SZ * (k1 + KERNEL_SZ * (l2 + LAYER3_AMT * l1));
                    end else if (step == 4) begin
                        l4_addr     <= k2 + x + LAYER4_SZ * (y + k1 + LAYER4_SZ * l1);
                        l5_addr     <= x + LAYER5_SZ * (y + LAYER5_SZ * l2);
                        b5_addr     <= l2;
                        w5_addr     <= k2 + KERNEL_SZ * (k1 + KERNEL_SZ * (l2 + LAYER5_AMT * l1));

                        
                    // Direcciones de submuestreo
                    end else if (step == 1) begin
                        l1_addr     <= k2 + x * DIM + LAYER1_SZ * (k1 + y * DIM + LAYER1_SZ * l2);
                        l2_addr     <= x + LAYER2_SZ * (y + LAYER2_SZ * l2);
                    end else if (step == 3) begin
                        l3_addr     <= k2 + x * DIM + LAYER3_SZ * (k1 + y * DIM + LAYER3_SZ * l2);
                        l4_addr     <= x + LAYER4_SZ * (y + LAYER4_SZ * l2);
                    
                    // Direcciones para el producto escalar:
                    end else begin
                        l5_addr     <= l1;
                        ol_addr     <= l2;
                        b6_addr     <= l2;
                        w6_addr     <= l2 + OUTPUT_AMT * l1;
                    end
                    
                    stage   <= FETCHING;
                
                // FASE DE LECTURA DE MEMORIA
                end else if (stage == FETCHING) begin
                    stage   <= EXECUTING;
                
                    // Adelantamiento de lecturas (1 posición de memoria)
                    if (step == 0) begin
                        il_addr     <= k2 + 1 + x + INPUT_SZ * (y + k1);
                        w1_addr     <= k2 + 1 + KERNEL_SZ * (k1 + KERNEL_SZ * (l2 + LAYER1_AMT * l1));
                    end else if (step == 2) begin
                        l2_addr     <= k2 + 1 + x + LAYER2_SZ * (y + k1 + LAYER2_SZ * l1);
                        w3_addr     <= k2 + 1 + KERNEL_SZ * (k1 + KERNEL_SZ * (l2 + LAYER3_AMT * l1));
                    end else if (step == 4) begin
                        l4_addr     <= k2 + 1 + x + LAYER4_SZ * (y + k1 + LAYER4_SZ * l1);
                        w5_addr     <= k2 + 1 + KERNEL_SZ * (k1 + KERNEL_SZ * (l2 + LAYER5_AMT * l1));
                    end else if (step == 1)
                        l1_addr     <= k2 + 1 + x * DIM + LAYER1_SZ * (k1 + y * DIM + LAYER1_SZ * l2);
                    else if (step == 3)
                        l3_addr     <= k2 + 1 + x * DIM + LAYER3_SZ * (k1 + y * DIM + LAYER3_SZ * l2);
                
                
                // FASE DE EJECUCIÓN
                end else if (stage == EXECUTING) begin
                
                    // Productos (fases de convolución y producto escalar)
                    if (step == 0 || step == 2 || step == 4 || step == 5) begin

						if (step == 0)
							mul   = il_dout * w1_dout;
						else if (step == 2)
							mul   = l2_dout * w3_dout;
						else if (step == 4)
							mul   = l4_dout * w5_dout;
						else
							mul   = l5_dout * w6_dout;
						
						aux     = aux + {mul[63], mul[46:16]};
                      
                    // Cálculo del máximo (submuestreo)
                    end else if (step == 1 || step == 3) begin
                    
                        if (step == 1) begin
                            if (k1 == 0 && k2 == 0)
                                aux <= l1_dout;
                            else if (l1_dout > aux)
                                aux <= l1_dout;
                        end else if (step == 3) begin
                            if (k1 == 0 && k2 == 0)
                                aux <= l3_dout;
                            else if (l3_dout > aux)
                                aux <= l3_dout;
                        end
                    end
                    
                    // Ajuste de los índices
                    if (k2 == LIMIT_K - 1) begin
                        if (k1 == LIMIT_K - 1) begin
                            stage <= ADDRESSING;
                            k2    <= LIMIT_K;
                        end else begin
                            k1    <= k1 + 1;
                            k2    <= 0;
                        end
                    end else
                        k2 <= k2 + 1;
                    
                    // Adelantamiento de lecturas (2 posiciones de memoria)
                    if (k2 >= LIMIT_K - 2) begin
                    
                        if (step == 0) begin
                            il_addr     <= k2 - LIMIT_K + 2 + x + INPUT_SZ * (y + k1 + 1);
                            w1_addr     <= k2 - LIMIT_K + 2 + KERNEL_SZ * (k1 + 1 + KERNEL_SZ * (l2 + LAYER1_AMT * l1));
                        end else if (step == 2) begin
                            l2_addr     <= k2 - LIMIT_K + 2 + x + LAYER2_SZ * (y + k1 + 1 + LAYER2_SZ * l1);
                            w3_addr     <= k2 - LIMIT_K + 2 + KERNEL_SZ * (k1 + 1 + KERNEL_SZ * (l2 + LAYER3_AMT * l1));
                        end else if (step == 4) begin
                            l4_addr     <= k2 - LIMIT_K + 2 + x + LAYER4_SZ * (y + k1 + 1 + LAYER4_SZ * l1);
                            w5_addr     <= k2 - LIMIT_K + 2 + KERNEL_SZ * (k1 + 1 + KERNEL_SZ * (l2 + LAYER5_AMT * l1));
                        end else if (step == 1)
                            l1_addr     <= k2 - LIMIT_K + 2 + x * DIM + LAYER1_SZ * (k1 + 1 + y * DIM + LAYER1_SZ * l2);
                        else if (step == 3)
                            l3_addr     <= k2 - LIMIT_K + 2 + x * DIM + LAYER3_SZ * (k1 + 1 + y * DIM + LAYER3_SZ * l2);
                            
                    end else begin

                        if (step == 0) begin
                            il_addr     <= k2 + 2 + x + INPUT_SZ * (y + k1);
                            w1_addr     <= k2 + 2 + KERNEL_SZ * (k1 + KERNEL_SZ * (l2 + LAYER1_AMT * l1));
                        end else if (step == 2) begin
                            l2_addr     <= k2 + 2 + x + LAYER2_SZ * (y + k1 + LAYER2_SZ * l1);
                            w3_addr     <= k2 + 2 + KERNEL_SZ * (k1 + KERNEL_SZ * (l2 + LAYER3_AMT * l1));
                        end else if (step == 4) begin
                            l4_addr     <= k2 + 2 + x + LAYER4_SZ * (y + k1 + LAYER4_SZ * l1);
                            w5_addr     <= k2 + 2 + KERNEL_SZ * (k1 + KERNEL_SZ * (l2 + LAYER5_AMT * l1));
                        end else if (step == 1)
                            l1_addr     <= k2 + 2 + x * DIM + LAYER1_SZ * (k1 + y * DIM + LAYER1_SZ * l2);
                        else if (step == 3)
                            l3_addr     <= k2 + 2 + x * DIM + LAYER3_SZ * (k1 + y * DIM + LAYER3_SZ * l2);
                    
                    end 
                end
            end
        end
    end
end
endmodule


