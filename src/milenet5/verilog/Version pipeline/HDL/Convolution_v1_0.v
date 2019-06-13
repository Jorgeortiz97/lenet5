
`timescale 1 ns / 1 ps

module SRAM #
(
    parameter ADDR_WIDTH = 8, DEPTH = 256
)
(
    input wire i_clk,
    input wire [ADDR_WIDTH-1:0] i_addr, 
    input wire i_write,
    input wire [31:0] i_data,
    output reg [31:0] o_data 
);
    
    reg [31:0] memory_array [0:DEPTH-1]; 
    
    always @ (posedge i_clk)
    begin
        if(i_write) begin
            memory_array[i_addr] <= i_data;
        end
        else begin
            o_data <= memory_array[i_addr];
        end     
    end
endmodule


`timescale 1ns / 1ps


module Convolution_v1_0_S00_AXI #
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
)
(

   // Auto generated code
    input wire  S_AXI_ACLK,
    input wire  S_AXI_ARESETN,
    input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_AWADDR,
    input wire [2 : 0] S_AXI_AWPROT,
    input wire  S_AXI_AWVALID,
    output wire  S_AXI_AWREADY,
    input wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_WDATA,
    input wire [(C_S_AXI_DATA_WIDTH/8)-1 : 0] S_AXI_WSTRB,
    input wire  S_AXI_WVALID,
    output wire  S_AXI_WREADY,
    output wire [1 : 0] S_AXI_BRESP,
    output wire  S_AXI_BVALID,
    input wire  S_AXI_BREADY,
    input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_ARADDR,
    input wire [2 : 0] S_AXI_ARPROT,
    input wire  S_AXI_ARVALID,
    output wire  S_AXI_ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_RDATA,
    output wire [1 : 0] S_AXI_RRESP,
    output wire  S_AXI_RVALID,
    input wire  S_AXI_RREADY
);


// Auto generated code
reg [C_S_AXI_ADDR_WIDTH-1 : 0] 	axi_awaddr;
reg  	axi_awready;
reg  	axi_wready;
reg [1 : 0] 	axi_bresp;
reg  	axi_bvalid;
reg [C_S_AXI_ADDR_WIDTH-1 : 0] 	axi_araddr;
reg  	axi_arready;
reg [C_S_AXI_DATA_WIDTH-1 : 0] 	axi_rdata;
reg [1 : 0] 	axi_rresp;
reg  	axi_rvalid;

localparam integer ADDR_LSB = (C_S_AXI_DATA_WIDTH/32) + 1;
localparam integer OPT_MEM_ADDR_BITS = 8;

wire	 slv_reg_rden;
wire	 slv_reg_wren;
integer	 b;
reg	 aw_en;
assign slv_reg_rden = axi_arready & S_AXI_ARVALID & ~axi_rvalid;




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
SRAM #(10, 1024)    input_layer(S_AXI_ACLK, il_addr, il_en, il_din, il_dout);
SRAM #(13, 4704)    layer1(S_AXI_ACLK, l1_addr, l1_en, l1_din, l1_dout);
SRAM #(11, 1176)    layer2(S_AXI_ACLK, l2_addr, l2_en, l2_din, l2_dout);
SRAM #(11, 1600)    layer3(S_AXI_ACLK, l3_addr, l3_en, l3_din, l3_dout);
SRAM #(9, 400)      layer4(S_AXI_ACLK, l4_addr, l4_en, l4_din, l4_dout);
SRAM #(7, 120)      layer5(S_AXI_ACLK, l5_addr, l5_en, l5_din, l5_dout);
SRAM #(4, 10)       output_layer(S_AXI_ACLK, ol_addr, ol_en, ol_din, ol_dout);


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
SRAM #(8, 150)          weight1(S_AXI_ACLK, w1_addr, w1_en, w1_din, w1_dout);
SRAM #(12, 2400)        weight3(S_AXI_ACLK, w3_addr, w3_en, w3_din, w3_dout);
SRAM #(16, 48000)       weight5(S_AXI_ACLK, w5_addr, w5_en, w5_din, w5_dout);
SRAM #(11, 1200)        weight6(S_AXI_ACLK, w6_addr, w6_en, w6_din, w6_dout);

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
SRAM #(3, 6)      bias1(S_AXI_ACLK, b1_addr, b1_en, b1_din, b1_dout);
SRAM #(4, 16)     bias3(S_AXI_ACLK, b3_addr, b3_en, b3_din, b3_dout);
SRAM #(7, 120)    bias5(S_AXI_ACLK, b5_addr, b5_en, b5_din, b5_dout);
SRAM #(4, 10)     bias6(S_AXI_ACLK, b6_addr, b6_en, b6_din, b6_dout);


// Memoria LutRAM
reg signed [63:0] lutram[0:31];

reg [31:0] step = 0, reset = 0;
reg [15:0] l1, l2, x, y, k1, k2;
reg [1:0] stage = 0;
reg signed [63:0] mul, tmp;
reg signed [31:0] aux = 0;

// Límites
reg [31:0] LIMIT_K, LIMIT_L1, LIMIT_XY, LIMIT_L2;


always @( posedge S_AXI_ACLK )
begin
    if ( S_AXI_ARESETN == 1'b0 )
        addr <= 0;
    else begin
        
        // Establecimiento de las direcciones de acceso
        il_addr     <= addr;
        l1_addr     <= addr;
        l2_addr     <= addr;
        l3_addr     <= addr;
        l4_addr     <= addr;
        l5_addr     <= addr;
        ol_addr     <= addr;
        
        w1_addr     <= addr;
        w3_addr     <= addr;
        w5_addr     <= addr;
        w6_addr     <= addr;
        
        b1_addr     <= addr;
        b3_addr     <= addr;
        b5_addr     <= addr;
        b6_addr     <= addr;
        
        // Activación del modo de escritura
        il_en   <= (slv_reg_wren && axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 9'h1);
        w1_en   <= (slv_reg_wren && axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 9'h11);
        w3_en   <= (slv_reg_wren && axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 9'h12);
        w5_en   <= (slv_reg_wren && axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 9'h13);
        w6_en   <= (slv_reg_wren && axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 9'h14);
        b1_en   <= (slv_reg_wren && axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 9'h15);
        b3_en   <= (slv_reg_wren && axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 9'h16);
        b5_en   <= (slv_reg_wren && axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 9'h17);
        b6_en   <= (slv_reg_wren && axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 9'h18);
        
        // Escritura activa:
        if (slv_reg_wren)
        begin
        case ( axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] )
        
            // Parámetros de E/S
            9'h0: for ( b = 0; b <= (C_S_AXI_DATA_WIDTH/8)-1; b = b+1 )
               if ( S_AXI_WSTRB[b] == 1 )
                   addr[(b*8) +: 8] <= S_AXI_WDATA[(b*8) +: 8];

            // Capa de entrada
            9'h1: for ( b = 0; b <= (C_S_AXI_DATA_WIDTH/8)-1; b = b+1 )
                if ( S_AXI_WSTRB[b] == 1 )
                    il_din[(b*8) +: 8] <= S_AXI_WDATA[(b*8) +: 8];
            
            // Pesos y valores de polarización
            9'h11: for ( b = 0; b <= (C_S_AXI_DATA_WIDTH/8)-1; b = b+1 )
                if ( S_AXI_WSTRB[b] == 1 )
                    w1_din[(b*8) +: 8] <= S_AXI_WDATA[(b*8) +: 8];
            9'h12: for ( b = 0; b <= (C_S_AXI_DATA_WIDTH/8)-1; b = b+1 )
                if ( S_AXI_WSTRB[b] == 1 )
                    w3_din[(b*8) +: 8] <= S_AXI_WDATA[(b*8) +: 8];
            9'h13: for ( b = 0; b <= (C_S_AXI_DATA_WIDTH/8)-1; b = b+1 )
                if ( S_AXI_WSTRB[b] == 1 )
                    w5_din[(b*8) +: 8] <= S_AXI_WDATA[(b*8) +: 8];

            9'h14: for ( b = 0; b <= (C_S_AXI_DATA_WIDTH/8)-1; b = b+1 )
                if ( S_AXI_WSTRB[b] == 1 )
                    w6_din[(b*8) +: 8] <= S_AXI_WDATA[(b*8) +: 8];
            9'h15: for ( b = 0; b <= (C_S_AXI_DATA_WIDTH/8)-1; b = b+1 )
                if ( S_AXI_WSTRB[b] == 1 )
                    b1_din[(b*8) +: 8] <= S_AXI_WDATA[(b*8) +: 8];
            9'h16: for ( b = 0; b <= (C_S_AXI_DATA_WIDTH/8)-1; b = b+1 )
                if ( S_AXI_WSTRB[b] == 1 )
                    b3_din[(b*8) +: 8] <= S_AXI_WDATA[(b*8) +: 8];
            9'h17: for ( b = 0; b <= (C_S_AXI_DATA_WIDTH/8)-1; b = b+1 )
                if ( S_AXI_WSTRB[b] == 1 )
                    b5_din[(b*8) +: 8] <= S_AXI_WDATA[(b*8) +: 8];
            9'h18: for ( b = 0; b <= (C_S_AXI_DATA_WIDTH/8)-1; b = b+1 )
                if ( S_AXI_WSTRB[b] == 1 )
                    b6_din[(b*8) +: 8] <= S_AXI_WDATA[(b*8) +: 8];
                    
            // Registros de control 
            9'h1A:for ( b = 0; b <= (C_S_AXI_DATA_WIDTH/8)-1; b = b+1 )
                if ( S_AXI_WSTRB[b] == 1 )
                    reset[(b*8) +: 8] <= S_AXI_WDATA[(b*8) +: 8];
                    
        endcase
        end else if ( S_AXI_ARESETN == 1'b0 )
            axi_rdata  <= 0;
            
        // Lectura activa:
        else begin
        
            if (slv_reg_rden)
            begin     
            case ( axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] )
                9'h0:       axi_rdata <= addr;
    
                9'h1:       axi_rdata <= il_dout;
                9'h2:       axi_rdata <= l1_dout;
                9'h3:       axi_rdata <= l2_dout;
                9'h4:       axi_rdata <= l3_dout;
                9'h5:       axi_rdata <= l4_dout;
                9'h6:       axi_rdata <= l5_dout;
                9'h7:       axi_rdata <= ol_dout;
                
                
                9'h11:      axi_rdata <= w1_dout;
                9'h12:      axi_rdata <= w3_dout;
                9'h13:      axi_rdata <= w5_dout;
                9'h14:      axi_rdata <= w6_dout;
                9'h15:      axi_rdata <= b1_dout;
                9'h16:      axi_rdata <= b3_dout;
                9'h17:      axi_rdata <= b5_dout;
                9'h18:      axi_rdata <= b6_dout;
                
                9'h1A:      axi_rdata <= reset;
                9'h1B:      axi_rdata <= step;
                
                9'h1C:      axi_rdata <= k2;
                9'h1D:      axi_rdata <= k1;
                9'h1E:      axi_rdata <= x;
                9'h1F:      axi_rdata <= y;
                
                9'h20:      axi_rdata <= l1;
                9'h21:      axi_rdata <= l2;
                
                default :   axi_rdata <= 32'hFFFFFFFF;
            endcase
            end
            

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
                    
                        stage   <= 0;
                        k2      <= 0;
                        k1      <= 0;
                        l1      <= 0;
                        x       <= 0;
                        y       <= 0;
                        l2      <= 0;
                        aux     <= 0;
                        
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
                            // RESULTADO LISTO
                        end
                        
                        // Establecemos el modo escritura para la capa correspondiente de salida:
                        l1_en       <= 0;
                        l2_en       <= (step == 0);
                        l3_en       <= (step == 1);
                        l4_en       <= (step == 2);
                        l5_en       <= (step == 3);
                        ol_en       <= (step == 4);
                        
                        
                    end else begin
               
                        // Control de los índices:
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
                                            l3_din      <= aux + b3_dout;
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
                            end else begin if (step == 5)
                                l5_addr     <= l1;
                                ol_addr     <= l2;
                                b6_addr     <= l2;
                                w6_addr     <= l2 + OUTPUT_AMT * l1;
                            end
                            
                            stage   <= FETCHING;
						
						// FASE DE LECTURA DE MEMORIA
                        end else if (stage == FETCHING)
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
                        else if (stage == EXECUTING) begin

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
                                
                                aux     <= aux + {mul[63], mul[46:16]};

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
    end
end


// Auto generated code

assign S_AXI_AWREADY	= axi_awready;
assign S_AXI_WREADY	= axi_wready;
assign S_AXI_BRESP	= axi_bresp;
assign S_AXI_BVALID	= axi_bvalid;
assign S_AXI_ARREADY	= axi_arready;
assign S_AXI_RDATA	= axi_rdata;
assign S_AXI_RRESP	= axi_rresp;
assign S_AXI_RVALID	= axi_rvalid;

always @( posedge S_AXI_ACLK )
begin
  if ( S_AXI_ARESETN == 1'b0 )
    begin
      axi_awready <= 1'b0;
      aw_en <= 1'b1;
    end 
  else
    begin    
      if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID && aw_en)
        begin
          axi_awready <= 1'b1;
          aw_en <= 1'b0;
        end
        else if (S_AXI_BREADY && axi_bvalid)
            begin
              aw_en <= 1'b1;
              axi_awready <= 1'b0;
            end
      else           
        begin
          axi_awready <= 1'b0;
        end
    end 
end       

always @( posedge S_AXI_ACLK )
begin
  if ( S_AXI_ARESETN == 1'b0 )
    begin
      axi_awaddr <= 0;
    end 
  else
    begin    
      if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID && aw_en)
        begin
          axi_awaddr <= S_AXI_AWADDR;
        end
    end 
end       


always @( posedge S_AXI_ACLK )
begin
  if ( S_AXI_ARESETN == 1'b0 )
    begin
      axi_wready <= 1'b0;
    end 
  else
    begin    
      if (~axi_wready && S_AXI_WVALID && S_AXI_AWVALID && aw_en )
        begin
          axi_wready <= 1'b1;
        end
      else
        begin
          axi_wready <= 1'b0;
        end
    end 
end       

assign slv_reg_wren = axi_wready && S_AXI_WVALID && axi_awready && S_AXI_AWVALID;

always @( posedge S_AXI_ACLK )
begin
  if ( S_AXI_ARESETN == 1'b0 )
    begin
      axi_bvalid  <= 0;
      axi_bresp   <= 2'b0;
    end 
  else
    begin    
      if (axi_awready && S_AXI_AWVALID && ~axi_bvalid && axi_wready && S_AXI_WVALID)
        begin
          axi_bvalid <= 1'b1;
          axi_bresp  <= 2'b0;
        end                  
      else
        begin
          if (S_AXI_BREADY && axi_bvalid)
            begin
              axi_bvalid <= 1'b0; 
            end  
        end
    end
end   

always @( posedge S_AXI_ACLK )
begin
  if ( S_AXI_ARESETN == 1'b0 )
    begin
      axi_arready <= 1'b0;
      axi_araddr  <= 32'b0;
    end 
  else
    begin    
      if (~axi_arready && S_AXI_ARVALID)
        begin
          axi_arready <= 1'b1;
          axi_araddr  <= S_AXI_ARADDR;
        end
      else
        begin
          axi_arready <= 1'b0;
        end
    end 
end       

always @( posedge S_AXI_ACLK )
begin
  if ( S_AXI_ARESETN == 1'b0 )
    begin
      axi_rvalid <= 0;
      axi_rresp  <= 0;
    end 
  else
    begin    
      if (axi_arready && S_AXI_ARVALID && ~axi_rvalid)
        begin
          axi_rvalid <= 1'b1;
          axi_rresp  <= 2'b0;
        end   
      else if (axi_rvalid && S_AXI_RREADY)
        begin
          axi_rvalid <= 1'b0;
        end                
    end
end    

endmodule

