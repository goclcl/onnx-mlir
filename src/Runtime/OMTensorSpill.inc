/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ OMTensorSpill.inc - TinyMo tensor spilling runtime ------===//
//
// onnx-tensor-spilling / onnx-tinymo-opt 패스가 삽입한 ONNXCustomOp
// (krnl.call)가 호출하는 함수들. 텐서를 로컬 스토리지(파일)로 내리고
// 되가져온다. 파일은 $OM_SPILL_DIR (기본 ".")의 om_spill_<id>.bin.
//
// krnl.call ABI: 출력 OMTensor*들이 먼저, 입력 OMTensor*들이 다음,
// 정수 속성(int64_t)이 속성 이름의 사전순으로 마지막에 온다.
//
//===----------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMTensor.h"

#ifdef __cplusplus
extern "C" {
#endif

static void omSpillPath(char *buf, size_t n, int64_t id) {
  const char *dir = getenv("OM_SPILL_DIR");
  if (!dir)
    dir = ".";
  snprintf(buf, n, "%s/om_spill_%lld.bin", dir, (long long)id);
}

/* 텐서를 파일로 내린다. 토큰(1바이트)은 spill→fetch 순서 보장용. */
void om_spill(OMTensor *token, OMTensor *input, int64_t spillId) {
  char path[512];
  omSpillPath(path, sizeof(path), spillId);
  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "om_spill: cannot open %s\n", path);
    abort();
  }
  int64_t bytes = omTensorGetBufferSize(input);
  if (fwrite(omTensorGetDataPtr(input), 1, (size_t)bytes, f) !=
      (size_t)bytes) {
    fprintf(stderr, "om_spill: short write to %s\n", path);
    abort();
  }
  fclose(f);
  ((char *)omTensorGetDataPtr(token))[0] = 1;
}

/* 파일에서 텐서를 되가져온다. */
void om_fetch(OMTensor *output, OMTensor *token, int64_t spillId) {
  (void)token;
  char path[512];
  omSpillPath(path, sizeof(path), spillId);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "om_fetch: cannot open %s\n", path);
    abort();
  }
  int64_t bytes = omTensorGetBufferSize(output);
  if (fread(omTensorGetDataPtr(output), 1, (size_t)bytes, f) !=
      (size_t)bytes) {
    fprintf(stderr, "om_fetch: short read from %s\n", path);
    abort();
  }
  fclose(f);
}

/*
 * fetch + 2입력 Concat 융합: fetched 텐서를 실체화하지 않고 concat 결과에
 * 바로 쓴다. axis 앞쪽 차원들을 outer로 접어, outer 조각마다
 * [operand0 행 | operand1 행]을 이어 붙인다. fetchPos = fetched 텐서가
 * concat의 몇 번째 입력이었는지(0|1). 파일은 victim의 연속 레이아웃이므로
 * outer 오름차순 순회로 순차 read 한다.
 */
void om_fetch_concat2(OMTensor *output, OMTensor *other, OMTensor *token,
    int64_t axis, int64_t fetchPos, int64_t spillId) {
  (void)token;
  char path[512];
  omSpillPath(path, sizeof(path), spillId);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "om_fetch_concat2: cannot open %s\n", path);
    abort();
  }

  const int64_t *outShape = omTensorGetShape(output);
  const int64_t *otherShape = omTensorGetShape(other);
  int64_t rank = omTensorGetRank(output);
  int64_t elemSize =
      omTensorGetBufferSize(output) / omTensorGetNumElems(output);

  int64_t outer = 1;
  for (int64_t d = 0; d < axis; ++d)
    outer *= outShape[d];
  int64_t inner = elemSize;
  for (int64_t d = axis + 1; d < rank; ++d)
    inner *= outShape[d];

  int64_t otherRow = otherShape[axis] * inner;
  int64_t fetchRow = (outShape[axis] - otherShape[axis]) * inner;

  char *out = (char *)omTensorGetDataPtr(output);
  const char *oth = (const char *)omTensorGetDataPtr(other);

  for (int64_t o = 0; o < outer; ++o) {
    char *dst = out + o * (otherRow + fetchRow);
    char *fetchDst = (fetchPos == 0) ? dst : dst + otherRow;
    char *otherDst = (fetchPos == 0) ? dst + fetchRow : dst;
    memcpy(otherDst, oth + o * otherRow, (size_t)otherRow);
    if (fread(fetchDst, 1, (size_t)fetchRow, f) != (size_t)fetchRow) {
      fprintf(stderr, "om_fetch_concat2: short read from %s\n", path);
      abort();
    }
  }
  fclose(f);
}

#ifdef __cplusplus
}
#endif
