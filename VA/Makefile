DPU_DIR := dpu
HOST_DIR := host
BUILDDIR ?= bin
NR_TASKLETS ?= 16
BL ?= 10
NR_DPUS ?= 1
TYPE ?= INT32
ENERGY ?= 0

define conf_filename
	${BUILDDIR}/.NR_DPUS_$(1)_NR_TASKLETS_$(2)_BL_$(3)_TYPE_$(4).conf
endef
CONF := $(call conf_filename,${NR_DPUS},${NR_TASKLETS},${BL},${TYPE})

HOST_TARGET := ${BUILDDIR}/host_code
DPU_TARGET := ${BUILDDIR}/dpu_code

COMMON_INCLUDES := support
HOST_SOURCES := $(wildcard ${HOST_DIR}/*.c)
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c)
# 와일드 카드 - 명령을 내릴 때, 여러 파일을 한꺼번에 지정할 목적으로 사용 (해당 디렉토리의 여러 .c sources 의미)

.PHONY: all clean test

__dirs := $(shell mkdir -p ${BUILDDIR}) // 디렉토리 생성

COMMON_FLAGS := -Wall -Wextra -g -I${COMMON_INCLUDES}
HOST_FLAGS := ${COMMON_FLAGS} -std=c11 -O3 `dpu-pkg-config --cflags --libs dpu` -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPUS=${NR_DPUS} -DBL=${BL} -D${TYPE} -DENERGY=${ENERGY}
DPU_FLAGS := ${COMMON_FLAGS} -O2 -DNR_TASKLETS=${NR_TASKLETS} -DBL=${BL} -D${TYPE}

all: ${HOST_TARGET} ${DPU_TARGET}

${CONF}:
	$(RM) $(call conf_filename,*,*)
	touch ${CONF}
	# 설정 파일을 생성, 이전 설정 파일 제거

# HOST Compile
${HOST_TARGET}: ${HOST_SOURCES} ${COMMON_INCLUDES} ${CONF}
	$(CC) -o $@ ${HOST_SOURCES} ${HOST_FLAGS}

# DPU Compile
${DPU_TARGET}: ${DPU_SOURCES} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES}

clean:
	$(RM) -r $(BUILDDIR)
	# build dir 삭제

test: all
	./${HOST_TARGET}
	# 빌드된 host 코드 실행
