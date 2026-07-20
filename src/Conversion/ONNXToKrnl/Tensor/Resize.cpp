/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Resize.cpp - Lowering Resize Op ---------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Resize Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXResizeOpLowering : public OpConversionPattern<ONNXResizeOp> {
  ONNXResizeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXResizeOp resizeOp,
      ONNXResizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    Operation *op = resizeOp.getOperation();
    Location loc = ONNXLoc<ONNXResizeOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value data = adaptor.getX();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);
    int64_t rank = memRefType.getShape().size();

    // Check limitation imposed by implementation
    // Resize Op is either lowered to loop nests or a function call.
    // In either case, only the default value for some of the attributes are
    // allowed.
    // In the library for Resize, src/Runtime/OMResize.inc, it seems that
    // it is easy to support some other values, but not tested.
    if (resizeOp.getAntialias() != 0 ||
        resizeOp.getCubicCoeffA().convertToDouble() != -0.75 ||
        resizeOp.getExcludeOutside() != 0 ||
        resizeOp.getExtrapolationValue().convertToDouble() != 0. ||
        resizeOp.getExcludeOutside() != 0 ||
        resizeOp.getKeepAspectRatioPolicy() != "stretch") {
      return emitError(
          loc, "attribute value not supported by current implementation#1");
    }

    // When getMode() is "nearest", Resize is lowered to loops.
    // getCoordinateTransformationMode() can be "asymmetric" or "half_pixel"
    if (resizeOp.getMode() == "nearest") {
      if (resizeOp.getCoordinateTransformationMode() != "asymmetric" &&
          resizeOp.getCoordinateTransformationMode() != "half_pixel") {
        return emitError(
            loc, "attribute value not supported by current implementation#2");
      }
    } else {
      // "linear" is lowered to loops and "cubic" to a library call, both
      // supporting only "half_pixel" for getCoordinateTransformationMode().
      if (resizeOp.getCoordinateTransformationMode() != "half_pixel") {
        return emitError(
            loc, "attribute value not supported by current implementation#3");
      }
    }

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // Shape helper: compute output dims and scales.
    ONNXResizeOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    Value alloc =
        create.mem.alignedAlloc(memRefType, shapeHelper.getOutputDims());

    // "nearest" (any element type) and "linear" (float element types) are
    // lowered to loop nests. Otherwise ("cubic", or "linear" on non-float
    // element types), call an external function.
    // Create KrnlCallOp and replace the du chain
    // One of inputs, getScales() and size(), has to be None.
    // For now, None input is picked out by KrnlCall builder,
    // and different function will be called accordingly.
    // Another issue is the attributes with default value.
    // Currently, it is assumed that all the optional attributes have
    // the default value and does appear in the Attribute dictionary.
    // ToFix: Handle attributes for general case
    // A list of attribute names for krnl.call is provided to determine which
    // attributes are passed to the call and in which order
    // The unspecified the attributes are assumed to have the default value.
    Type elementType = memRefType.getElementType();
    bool genLinearCode = resizeOp.getMode() == "linear" &&
                         mlir::isa<FloatType>(elementType);
    if (resizeOp.getMode() != "nearest" && !genLinearCode) {
      std::vector<std::string> attributeNames = {"mode", "nearest_mode"};
      if (!isNoneValue(resizeOp.getScales())) {
        rewriter.create<KrnlCallOp>(
            loc, "Resize_Scales", alloc, op, operands, attributeNames);
      } else {
        rewriter.create<KrnlCallOp>(
            loc, "Resize_Size", alloc, op, operands, attributeNames);
      }
      rewriter.replaceOp(op, alloc);
      onnxToKrnlSimdReport(op);
      return success();
    }
    // It is much more efficient to generate codes directly if possible

    SmallVector<Value, 4> scaleValues;
    IndexExpr::getValues(shapeHelper.scales, scaleValues);

    // Constants used in the loop body
    Value zero = create.math.constant(rewriter.getIntegerType(64), 0);
    Value one = create.math.constantIndex(1);

    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr, 4> lbs(rank, LitIE(0));
    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(alloc, ubs);
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](const KrnlBuilder &ck, ValueRange loopInd) {
          MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder>
              create(ck);

          // Clamp an int64 index into [0, dims(data)[i] - 1] and cast it to
          // an index value. When the index is out of bound, use the boundary
          // index. This is equivalent to np.pad with mode = "edge".
          auto clampToInputBound = [&](Value inIndexInteger,
                                       int64_t i) -> Value {
            // Compare with integer type because lower bound may be negative
            Value lessThanZero = create.math.slt(inIndexInteger, zero);
            Value inIndexLBPadded =
                create.math.select(lessThanZero, zero, inIndexInteger);

            // Upper bound comparison can be done with Index type
            inIndexLBPadded = create.math.castToIndex(inIndexLBPadded);
            Value inputDim = create.krnlIE.getShapeAsDim(data, i).getValue();

            Value lessThanDim = create.math.slt(inIndexLBPadded, inputDim);
            Value inputDimMinus = create.math.sub(inputDim, one);
            return create.math.select(
                lessThanDim, inIndexLBPadded, inputDimMinus);
          };

          if (genLinearCode) {
            // Multi-linear interpolation. Along each axis, the value at the
            // original (float) coordinate is the weighted sum of the two
            // neighboring input elements. The output value accumulates the
            // products of the per-axis weights over the cartesian product of
            // the neighbor indices. An axis whose scale is 1.0 at compile
            // time maps identically to the input ("half_pixel" gives
            // x_original = (x_resized + 0.5) / 1.0 - 0.5 = x_resized) and
            // needs no interpolation.
            Type f32Type = rewriter.getF32Type();
            Type i64Type = rewriter.getIntegerType(64);
            // Accumulate in f64 when the data is f64, in f32 otherwise.
            Type computeType = elementType.isF64() ? (Type)rewriter.getF64Type()
                                                   : (Type)f32Type;
            Value halfPixelConstant = create.math.constant(f32Type, 0.5);
            Value oneFloat = create.math.constant(f32Type, 1.0);
            Value oneInteger = create.math.constant(i64Type, 1);

            // For each axis: the (index, weight) pair of each neighbor.
            // Identity axes have a single neighbor with a null weight.
            SmallVector<SmallVector<std::pair<Value, Value>, 2>, 4> neighbors;
            for (int64_t i = 0; i < rank; ++i) {
              if (shapeHelper.scales[i].isLiteralAndIdenticalTo(1.0)) {
                neighbors.push_back({{loopInd[i], nullptr}});
                continue;
              }
              Value outIndexInteger = rewriter.create<arith::IndexCastOp>(
                  loc, i64Type, loopInd[i]);
              Value outIndexFloat = create.math.cast(f32Type, outIndexInteger);
              // Only "half_pixel" is supported for "linear" (checked above):
              // x_original = (x_resized + 0.5) / scale - 0.5.
              Value addValue =
                  create.math.add(outIndexFloat, halfPixelConstant);
              Value divValue = create.math.div(addValue, scaleValues[i]);
              Value inIndexFloat = create.math.sub(divValue, halfPixelConstant);
              Value inIndexFloor = create.math.floor(inIndexFloat);
              Value ratio = create.math.sub(inIndexFloat, inIndexFloor);
              // FPToSIOp is exact here because inIndexFloor is integral.
              Value inIndexInteger = create.math.cast(i64Type, inIndexFloor);
              Value inIndexP1Integer =
                  create.math.add(inIndexInteger, oneInteger);
              Value index0 = clampToInputBound(inIndexInteger, i);
              Value index1 = clampToInputBound(inIndexP1Integer, i);
              Value weight1 = ratio;
              Value weight0 = create.math.sub(oneFloat, ratio);
              if (computeType != f32Type) {
                weight0 = create.math.cast(computeType, weight0);
                weight1 = create.math.cast(computeType, weight1);
              }
              neighbors.push_back({{index0, weight0}, {index1, weight1}});
            }

            // Accumulate the weighted neighbor values over the cartesian
            // product of the per-axis neighbors (2^(number of interpolated
            // axes) terms).
            Value sum = nullptr;
            SmallVector<size_t, 4> pick(rank, 0);
            while (true) {
              SmallVector<Value, 4> readIndices;
              Value weight = nullptr;
              for (int64_t i = 0; i < rank; ++i) {
                const auto &neighbor = neighbors[i][pick[i]];
                readIndices.emplace_back(neighbor.first);
                if (neighbor.second)
                  weight = weight ? create.math.mul(weight, neighbor.second)
                                  : neighbor.second;
              }
              Value loadVal = create.krnl.load(data, readIndices);
              if (elementType != computeType)
                loadVal = create.math.cast(computeType, loadVal);
              Value term = weight ? create.math.mul(weight, loadVal) : loadVal;
              sum = sum ? create.math.add(sum, term) : term;
              // Advance to the next combination of neighbors.
              int64_t axis = rank - 1;
              while (axis >= 0 && ++pick[axis] == neighbors[axis].size()) {
                pick[axis] = 0;
                --axis;
              }
              if (axis < 0)
                break;
            }
            if (elementType != computeType)
              sum = create.math.cast(elementType, sum);
            create.krnl.store(sum, alloc, loopInd);
            return;
          }

          // mode == "nearest"
          SmallVector<Value, 4> readIndices;
          for (int64_t i = 0; i < rank; ++i) {
            Value inIndexFloat;
            Value outIndex = loopInd[i];
            Value outIndexInteger = rewriter.create<arith::IndexCastOp>(
                loc, rewriter.getIntegerType(64), outIndex);
            Value outIndexFloat =
                create.math.cast(rewriter.getF32Type(), outIndexInteger);

            // Handle coordinate transformation
            if (resizeOp.getCoordinateTransformationMode() == "asymmetric") {
              inIndexFloat = create.math.div(outIndexFloat, scaleValues[i]);
            } else if (resizeOp.getCoordinateTransformationMode() ==
                       "half_pixel") {
              // If coordinate_transformation_mode is "half_pixel",
              // x_original = (x_resized + 0.5) / scale - 0.5,
              Value halfPixelConstant =
                  create.math.constant(rewriter.getF32Type(), 0.5);
              Value addValue =
                  create.math.add(outIndexFloat, halfPixelConstant);
              Value divValue = create.math.div(addValue, scaleValues[i]);
              inIndexFloat = create.math.sub(divValue, halfPixelConstant);
            }

            // Handle nearest_mode
            if (resizeOp.getNearestMode() == "round_prefer_floor") {
              // round_prefer_floor will round 2.5 to 2, not 3
              Value deltaConstant =
                  create.math.constant(rewriter.getF32Type(), 0.499999);
              inIndexFloat = create.math.add(inIndexFloat, deltaConstant);
            } else if (resizeOp.getNearestMode() == "round_prefer_ceil") {
              Value deltaConstant =
                  create.math.constant(rewriter.getF32Type(), 0.5);
              inIndexFloat = create.math.add(inIndexFloat, deltaConstant);
            } else if (resizeOp.getNearestMode() == "floor") {
              // Not supported by create.math
              inIndexFloat = rewriter.create<math::FloorOp>(loc, inIndexFloat);
            } else if (resizeOp.getNearestMode() == "ceil") {
              // Not supported by create.math
              inIndexFloat = rewriter.create<math::CeilOp>(loc, inIndexFloat);
            } else {
              llvm_unreachable("Unexpected getNearestMode() for ResizeOp");
            }

            // FPToSIOp is round-to-zero, same as floor for positive
            Value inIndexInteger =
                create.math.cast(rewriter.getIntegerType(64), inIndexFloat);
            readIndices.emplace_back(clampToInputBound(inIndexInteger, i));
          }
          Value loadVal = create.krnl.load(data, readIndices);
          create.krnl.store(loadVal, alloc, loopInd);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXResizeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXResizeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
