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
 * fetch + Concat 융합 (2~4입력): fetched 텐서를 실체화하지 않고 concat
 * 결과에 바로 쓴다. axis 앞쪽 차원들을 outer로 접어, outer 조각마다
 * [part0 | part1 | ...] 을 이어 붙인다. fetchPos 파트는 파일에서 순차
 * read하고(파일 = victim의 연속 레이아웃, outer 오름차순 순회와 일치),
 * 나머지 파트는 others[k]에서 복사한다. others = 원래 concat 순서에서
 * victim을 뺀 순서.
 */
static void omFetchConcatImpl(OMTensor *out, OMTensor **others,
    int64_t nOthers, int64_t axis, int64_t fetchPos, int64_t spillId) {
  char path[512];
  omSpillPath(path, sizeof(path), spillId);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "om_fetch_concat: cannot open %s\n", path);
    abort();
  }

  const int64_t *outShape = omTensorGetShape(out);
  int64_t rank = omTensorGetRank(out);
  int64_t elemSize = omTensorGetBufferSize(out) / omTensorGetNumElems(out);

  int64_t outer = 1;
  for (int64_t d = 0; d < axis; ++d)
    outer *= outShape[d];
  int64_t inner = elemSize;
  for (int64_t d = axis + 1; d < rank; ++d)
    inner *= outShape[d];

  int64_t nParts = nOthers + 1;
  int64_t widths[4];
  int64_t sumOthers = 0;
  for (int64_t p = 0, oi = 0; p < nParts; ++p) {
    if (p == fetchPos)
      continue;
    widths[p] = omTensorGetShape(others[oi])[axis] * inner;
    sumOthers += widths[p];
    ++oi;
  }
  int64_t rowTotal = outShape[axis] * inner;
  widths[fetchPos] = rowTotal - sumOthers;

  char *outp = (char *)omTensorGetDataPtr(out);
  for (int64_t o = 0; o < outer; ++o) {
    char *dst = outp + o * rowTotal;
    int64_t off = 0;
    for (int64_t p = 0, oi = 0; p < nParts; ++p) {
      if (p == fetchPos) {
        if (fread(dst + off, 1, (size_t)widths[p], f) != (size_t)widths[p]) {
          fprintf(stderr, "om_fetch_concat: short read from %s\n", path);
          abort();
        }
      } else {
        const char *src = (const char *)omTensorGetDataPtr(others[oi]);
        memcpy(dst + off, src + o * widths[p], (size_t)widths[p]);
        ++oi;
      }
      off += widths[p];
    }
  }
  fclose(f);
}

void om_fetch_concat2(OMTensor *out, OMTensor *o1, OMTensor *token,
    int64_t axis, int64_t fetchPos, int64_t spillId) {
  (void)token;
  OMTensor *others[1] = {o1};
  omFetchConcatImpl(out, others, 1, axis, fetchPos, spillId);
}

void om_fetch_concat3(OMTensor *out, OMTensor *o1, OMTensor *o2,
    OMTensor *token, int64_t axis, int64_t fetchPos, int64_t spillId) {
  (void)token;
  OMTensor *others[2] = {o1, o2};
  omFetchConcatImpl(out, others, 2, axis, fetchPos, spillId);
}

void om_fetch_concat4(OMTensor *out, OMTensor *o1, OMTensor *o2, OMTensor *o3,
    OMTensor *token, int64_t axis, int64_t fetchPos, int64_t spillId) {
  (void)token;
  OMTensor *others[3] = {o1, o2, o3};
  omFetchConcatImpl(out, others, 3, axis, fetchPos, spillId);
}

#ifdef __cplusplus
}
#endif
