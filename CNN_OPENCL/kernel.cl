

__kernel void convhelper(
    __global float* input,
    const int NBYN,
    __global float* IMG2COL,
    __local float* LOCAL_INPUT,
    const int inDim
) {
    const int GLOBAL_ROW = get_global_id(1);
    const int GLOBAL_COL = get_global_id(0);
    const int IMGNUM = get_global_id(2);

    const int LOCAL_ROW = get_local_id(1);
    const int LOCAL_COL = get_local_id(0);

    const int FACE_X = LOCAL_COL % NBYN;
    const int FACE_Y = LOCAL_COL / NBYN;

    LOCAL_INPUT[LOCAL_COL] = input[(NBYN * NBYN * inDim * IMGNUM) + (NBYN * NBYN * GLOBAL_ROW) + GLOBAL_COL];

    barrier(CLK_LOCAL_MEM_FENCE);
    const int IMG2COL_BASE = (NBYN * NBYN * 9 * inDim * IMGNUM);

    for (int FILTER_ROW = 0; FILTER_ROW < 3; FILTER_ROW++) {
        for (int FILTER_COL = 0; FILTER_COL < 3; FILTER_COL++) {
            int x = FACE_X + FILTER_COL - 1;
            int y = FACE_Y + FILTER_ROW - 1;

            IMG2COL[IMG2COL_BASE + ((NBYN * NBYN) * (GLOBAL_ROW * 9 + FILTER_ROW * 3 + FILTER_COL)) + (GLOBAL_COL)] = (x >= 0 && x < NBYN && y >= 0 && y < NBYN) ? LOCAL_INPUT[(NBYN * y) + (x)] : 0.0f;
        }
    }
}



#define WPT4 16
__kernel void convt(
    __global float* input,
    __global float* weight,
    __global float* bias,
    __global float* output,
    const int inDim,
    const int outDim,
    const int NBYN,
    __local float* Asub,
    __local float* Bsub,
    const int TS,
    int img_num_per_ch  
) {
   
    const int LOCAL_ROW = get_local_id(1);
    const int LOCAL_COL = get_local_id(0);

    const int IMGNUM = get_global_id(2)*img_num_per_ch;
    
    const int RTS = (TS / WPT4);
    float sum[WPT4];
    for (int w = 0; w < WPT4; w++) {
        sum[w] = 0.0f;
    }

   const int GLOBAL_ROW = get_group_id(1) * TS + LOCAL_ROW;//gy
   int GLOBAL_COL = get_group_id(0) * TS + LOCAL_COL;//gx
   int lx = get_global_id(0);

   //float sum = 0.0f;
   const int COL_A = 9 * inDim; 
   const int COL_B = NBYN * NBYN;

   
   const int ROW_A = outDim;
   int img_size = NBYN * NBYN;

   for (int t = 0; t < COL_A; t += TS) {
        for(int w=0;w<WPT4;w++){
            const int TILE_ROW = t + LOCAL_ROW;//global에 접근할 인덱스
            const int TILE_COL = t + LOCAL_COL;//global에 접근할 인덱스

            //협력해서 글로벌에서 로컬로 값을 넣는 단계
            Asub[(LOCAL_ROW + w * 2) * TS + LOCAL_COL] = (TILE_COL < COL_A) ? weight[(COL_A * (GLOBAL_ROW + w * 2)) + (TILE_COL)] : 0.0f;          
            Bsub[(LOCAL_ROW + w * 2) * TS + LOCAL_COL] = (TILE_ROW < COL_A ) ? input[(COL_A * COL_B * (IMGNUM+(GLOBAL_COL/4))) + (COL_B * (TILE_ROW + w * 2)) + GLOBAL_COL%4] : 0.0f;
               
       
       }
       barrier(CLK_LOCAL_MEM_FENCE);

       //행렬곱 중간 결과 누적단계
       
       for (int k = 0; k < TS; k++) {
            for (int w = 0; w < WPT4; w++) {
                sum[w] += Asub[(LOCAL_ROW + w * 2) * TS + k] * Bsub[k * TS + LOCAL_COL];
            }
        }
       barrier(CLK_LOCAL_MEM_FENCE);
   }
   
    for (int w = 0; w < WPT4; w++) {
        output[(ROW_A * COL_B * (IMGNUM+(GLOBAL_COL/4))) + (COL_B * (GLOBAL_ROW + w * 2)) + (GLOBAL_COL%4)] = fmax(0,sum[w] + bias[(GLOBAL_ROW + w * 2)]);
     }
    
  

  
    
}

#define WPT5 16
__kernel void convt2(
    __global float* input,
    __global float* weight,
    __global float* bias,
    __global float* output,
    const int inDim,
    const int outDim,
    const int NBYN,
    __local float* Asub,
    __local float* Bsub,
    const int TS,
    int img_num_per_ch  
) {
    
    
    const int LOCAL_ROW = get_local_id(1);
    const int LOCAL_COL = get_local_id(0);

    const int IMGNUM = get_global_id(2)*img_num_per_ch;
    
    const int RTS = (TS / WPT5);
    float sum[WPT5];
    for (int w = 0; w < WPT5; w++) {
        sum[w] = 0.0f;
    }

   const int GLOBAL_ROW = get_group_id(1) * TS + LOCAL_ROW;//gy
   int GLOBAL_COL = get_group_id(0) * TS + LOCAL_COL;//gx
   int lx = get_global_id(0);

   //float sum = 0.0f;
   const int COL_A = 9 * inDim; 
   const int COL_B = NBYN * NBYN;

   
   const int ROW_A = outDim;
   int img_size = NBYN * NBYN;

   for (int t = 0; t < COL_A; t += TS) {
        for(int w=0;w<WPT5;w++){
            const int TILE_ROW = t + LOCAL_ROW;//global에 접근할 인덱스
            const int TILE_COL = t + LOCAL_COL;//global에 접근할 인덱스

            //협력해서 글로벌에서 로컬로 값을 넣는 단계
            
            Asub[(LOCAL_ROW + w * RTS) * TS + LOCAL_COL] = weight[(COL_A * (GLOBAL_ROW + w * RTS)) + (TILE_COL)];          
            Bsub[(LOCAL_ROW + w * RTS) * TS + LOCAL_COL] = input[(COL_A * 16 * (IMGNUM+(GLOBAL_COL/16))) + (16 * (TILE_ROW + w * RTS)) + GLOBAL_COL%16];
       }
       barrier(CLK_LOCAL_MEM_FENCE);

       //행렬곱 중간 결과 누적단계
       
       for (int k = 0; k < TS; k++) {
            for (int w = 0; w < WPT5; w++) {
                sum[w] += Asub[(LOCAL_ROW + w * RTS) * TS + k] * Bsub[k * TS + LOCAL_COL];
            }
        }
       barrier(CLK_LOCAL_MEM_FENCE);
   }
   
    for (int w = 0; w < WPT5; w++) {
        output[(ROW_A * COL_B * (IMGNUM+(GLOBAL_COL/16))) + (16 * (GLOBAL_ROW + w * RTS)) + (GLOBAL_COL%16)] = fmax(0,sum[w] + bias[(GLOBAL_ROW + w * RTS)]);
     
  }
    
}


#define WPT 16
__kernel void convhighperf(
    __global float* input,
    __global float* weight,
    __global float* bias,
    __global float* output,
    const int inDim,
    const int outDim,
    const int NBYN,
    __local float* Asub,
    __local float* Bsub,
    const int TS
) {
    const int LOCAL_ROW = get_local_id(1);
    const int LOCAL_COL = get_local_id(0);
    const int IMGNUM = get_global_id(2);

    const int GLOBAL_ROW = get_group_id(1) * TS + LOCAL_ROW;
    const int GLOBAL_COL = get_group_id(0) * TS + LOCAL_COL;

    

    float sum[WPT];
    for (int w = 0; w < WPT; w++) {
        sum[w] = 0.0f;
    }
    const int COL_A = 9 * inDim; 
    const int COL_B = NBYN * NBYN;
    const int ROW_A = outDim;
    
    for (int t = 0; t < COL_A; t += TS) {
        for (int w = 0; w < WPT; w++) {
            const int TILE_ROW = t + LOCAL_ROW;
            const int TILE_COL = t + LOCAL_COL;           
            Asub[(LOCAL_ROW + w * 4) * TS + LOCAL_COL] = weight[(COL_A * (GLOBAL_ROW + w * 4)) + (TILE_COL)];
            Bsub[(LOCAL_ROW + w * 4) * TS + LOCAL_COL] = input[(COL_A * COL_B * IMGNUM) + (COL_B * (TILE_ROW + w * 4)) + GLOBAL_COL];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            for (int w = 0; w < WPT; w++) {
                sum[w] += Asub[(LOCAL_ROW + w * 4) * TS + k] * Bsub[k * TS + LOCAL_COL];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WPT; w++) {
        output[(ROW_A * COL_B * IMGNUM) + (COL_B * (GLOBAL_ROW + w * 4)) + GLOBAL_COL] = fmax(0,sum[w] + bias[(GLOBAL_ROW + w * 4)]);
    }
}

#define WPT3 16
__kernel void convhighperf_first(
    __global float* input,
    __global float* weight,
    __global float* bias,
    __global float* output,
    const int inDim,
    const int outDim,
    const int NBYN,
    __local float* Asub,
    __local float* Bsub,
    const int TS
) {
    const int LOCAL_ROW = get_local_id(1);
    const int LOCAL_COL = get_local_id(0);
    const int IMGNUM = get_global_id(2);

    const int GLOBAL_ROW = get_group_id(1) * TS + LOCAL_ROW;
    const int GLOBAL_COL = get_group_id(0) * TS + LOCAL_COL;

    const int RTS = (TS / WPT3);

    float sum[WPT3];
    for (int w = 0; w < WPT; w++) {
        sum[w] = 0.0f;
    }
    const int COL_A = 9 * inDim; // 9 x 64 = 576
    const int COL_B = NBYN * NBYN; // 32 x 32 = 1024
    const int ROW_A = outDim;
    
    for (int t = 0; t < COL_A; t += TS) {
        for (int w = 0; w < WPT3; w++) {
            const int TILE_ROW = t + LOCAL_ROW;
            const int TILE_COL = t + LOCAL_COL;

            Asub[(LOCAL_ROW + w * 4) * TS + LOCAL_COL] = (TILE_COL < COL_A) ? weight[(COL_A * (GLOBAL_ROW + w * 4)) + (TILE_COL)] : 0.0f;
            Bsub[(LOCAL_ROW + w * 4) * TS + LOCAL_COL] = (TILE_ROW < COL_A) ? input[(COL_A * COL_B * IMGNUM) + (COL_B * (TILE_ROW + w * 4)) + GLOBAL_COL] : 0.0f;
           
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            for (int w = 0; w < WPT3; w++) {
                sum[w] += Asub[(LOCAL_ROW + w * 4) * TS + k] * Bsub[k * TS + LOCAL_COL];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WPT3; w++) {
        output[(ROW_A * COL_B * IMGNUM) + (COL_B * (GLOBAL_ROW + w * 4)) + GLOBAL_COL] = fmax(0,sum[w] + bias[(GLOBAL_ROW + w * 4)]);
    }
}


__kernel void maxPooling(
    __global float* input,
    __global float* output,
    const int inDim,
    const int NBYN
) {
    const int IMGNUM = get_global_id(2);
    const int LOCALID = get_local_id(1);
    const int FACE_LOCAL_X = get_global_id(0) % NBYN;
    const int FACE_LOCAL_Y = get_global_id(0) / NBYN;
    const int OUT_CH_ID = get_group_id(1);

    const int OUTIDX_BASE = (inDim * NBYN * NBYN * IMGNUM) + (NBYN * NBYN * OUT_CH_ID) + (NBYN * FACE_LOCAL_Y) + FACE_LOCAL_X;
    const int INPUTIDX_BASE = (inDim * (NBYN * 2) * (NBYN * 2) * IMGNUM) + ((NBYN * 2) * (NBYN * 2) * OUT_CH_ID);
    output[OUTIDX_BASE] = 0;

    for (int FILTER_ROW = 0; FILTER_ROW < 2; FILTER_ROW++) {
        for (int FILTER_COL = 0; FILTER_COL < 2; FILTER_COL++) {
            int x = (2 * FACE_LOCAL_X) + FILTER_COL;
            int y = (2 * FACE_LOCAL_Y) + FILTER_ROW;
            output[OUTIDX_BASE] = fmax(output[OUTIDX_BASE], input[INPUTIDX_BASE + ((NBYN * 2) * y) + (x)]);
        }
    }
}

__kernel void fcLayer(
    __global float* input,
    __global float* output,
    __global float* weight,
    __global float* bias,
    const int inDim,
    const int outDim,
    const int NBYN,
    __local float* LOCAL_SUM
) {
    const int IMGNUM = get_global_id(2);
    const int LOCALID = get_local_id(1);
    const int OUT_CH_ID = get_group_id(0);

    LOCAL_SUM[LOCALID] = input[(inDim * IMGNUM) + LOCALID] * weight[(inDim * OUT_CH_ID) + LOCALID];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int p = get_local_size(1) / 2; p >= 1; p = p >> 1) {
        if (LOCALID < p) LOCAL_SUM[LOCALID] += LOCAL_SUM[LOCALID + p];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (LOCALID == 0) {
        output[(outDim * IMGNUM) + OUT_CH_ID] = fmax(0, LOCAL_SUM[0] + bias[OUT_CH_ID]);
    }
}