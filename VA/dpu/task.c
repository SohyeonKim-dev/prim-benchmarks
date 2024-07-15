/*
* Vector addition with multiple tasklets
*
*/
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>

#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

// vector_addition: Computes the vector addition of a cached block 
// 벡터 덧셈 연산 at DPU
static void vector_addition(T *bufferB, T *bufferA, unsigned int l_size) {
    for (unsigned int i = 0; i < l_size; i++){
        bufferB[i] += bufferA[i];
    }
}

// Barrier - pthread - 스레드와 관련된 베리어 개념
// NR_TASKLETS개의 tasklet, 즉 스레드들의 동기화를 기다림 -> 이후 과정 진행 
BARRIER_INIT(my_barrier, NR_TASKLETS);

// 메인 함수를 가져온다는 의미? -> 외부에서 함수가 연결된다는 의미
extern int main_kernel1(void);

// 함수 포인터 배열을 초기화
// nr_kernels개 요소를 갖는 kernels array
// main_kernel1 함수가 first data 
// kernels array에 여러 커널 함수를 정의하여, 동적으로 호출
int (*kernels[nr_kernels])(void) = {main_kernel1};

int main(void) { 
    // kernel을 통해 kernels array의 여러 커널 함수를 호출할 수 있음! wow
    return kernels[DPU_INPUT_ARGUMENTS.kernel](); 
}

// main_kernel1
int main_kernel1() {
    unsigned int tasklet_id = me(); // 자기 자신의 id를 가져오기
    // 각각의 tasklet들은 고유한 id를 갖는다 

#if PRINT
    printf("tasklet_id = %u\n", tasklet_id);
#endif
    // print 구문을 매크로로 주는구나..!

    if (tasklet_id == 0){ // Initialize once the cycle counter
        mem_reset(); // Reset the heap
    }
    // Barrier - 다른 모든 tasklets 도달을 기다림 
    barrier_wait(&my_barrier);

    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size; // Input size per DPU in bytes
    uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.transfer_size; // Transfer input size per DPU in bytes

    // Address of the current processing block in MRAM -> MRAM에서 input 데이터를 저장할 base address 계산
    uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2; // shift 개념
    uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size_dpu_bytes_transfer);

    // Initialize a local cache to store the MRAM block 
    T *cache_A = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_B = (T *) mem_alloc(BLOCK_SIZE);

    for(unsigned int byte_index = base_tasklet; byte_index < input_size_dpu_bytes; byte_index += BLOCK_SIZE * NR_TASKLETS){

        // Bound checking
        uint32_t l_size_bytes = (byte_index + BLOCK_SIZE >= input_size_dpu_bytes) ? (input_size_dpu_bytes - byte_index) : BLOCK_SIZE;

        // Load cache with current MRAM block
        mram_read((__mram_ptr void const*)(mram_base_addr_A + byte_index), cache_A, l_size_bytes);
        mram_read((__mram_ptr void const*)(mram_base_addr_B + byte_index), cache_B, l_size_bytes);

        // Computer vector addition - 벡터 덧셈 연산 at DPU 
        // l_size = l_size_bytes >> DIV
        // DIV = 데이터 타입 별로 정의된 size -> 얘로 나누어서, add 할 개수를 구하는 것 
        vector_addition(cache_B, cache_A, l_size_bytes >> DIV);

        // Write cache to current MRAM block - 계산 결과를 MRAM에 write하는 과정 
        mram_write(cache_B, (__mram_ptr void*)(mram_base_addr_B + byte_index), l_size_bytes);

    }

    return 0;
}
