package com.example.tornadovm;

import com.example.core.model.GGMLType;

import com.example.core.types.Float16;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat4;
import uk.ac.manchester.tornado.api.types.vectors.Float4;

import java.util.stream.IntStream;

public class TornadoVMCompute {
    public static final boolean TORNADOVM = Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"));

    public TornadoVMCompute() {
    }

    public static void rmsnorm(FloatArray output, FloatArray input, FloatArray weights, IntArray positionAndLayer,
            int size, float ermsNorm) {
        // Calculate layer offset - weights for this layer start at this offset
        int layerOffset = positionAndLayer.get(1) * size;

        // Calculate sum of squares
        float sumSquares = 0.0f;
        for (int j = 0; j < size; j++) {
            sumSquares += input.get(j) * input.get(j);
        }
        sumSquares /= size;
        sumSquares += ermsNorm; // Add epsilon for numerical stability
        float scale = 1.0f / (float)TornadoMath.sqrt(sumSquares);

        // Normalize and scale with weights from the correct layer
        for (int j = 0; j < size; j++) {
            output.set(j,weights.get(layerOffset + j) * (scale * input.get(j)));
        }
    }
    public static void reductionOneBlockWithLayer(KernelContext context, FloatArray output, FloatArray x,
         int size, float ermsNorm, int localMemSize) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupId = context.groupIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory with the provided size
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        // Load input value and compute square
        if (gid < size) {
            localX[lid] = x.get(gid);
            localX[lid] = localX[lid] * localX[lid];
        } else {
            localX[lid] = 0.0f;
        }

        // Perform parallel reduction within the work group
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        // Each workgroup stores its partial sum in a different location
        if (lid == 0) {
            // Store the partial sum from each workgroup
            output.set(groupId + 1, localX[0]);
        }

        // Only the first thread in the first workgroup computes the final normalization factor
        if (gid == 0) {
            // Combine partial sums from all workgroups
            float ss = 0.0f;
            for (int i = 1; i <= 8; i++) {  // Assuming 8 workgroups
                ss += output.get(i);
            }

            ss /= size;
            ss += ermsNorm;
            ss = 1.0f / TornadoMath.sqrt(ss);
            output.set(0, ss);  // Store the final scale factor
        }
    }


    public static void initTempToZero(FloatArray temp, FloatArray tempFFN) {
        // Zero out all elements (even though in this case we only need first 9 elements)
        for (@Parallel int i = 0; i < temp.getSize(); i++) {
            temp.set(i, 0.0f);
            tempFFN.set(i, 0.0f);
        }
    }

    public static void reductionOneBlockWithLayer(KernelContext context, FloatArray output, FloatArray x,
            IntArray positionAndLayer, int size, float ermsNorm) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupSize = context.localGroupSizeX;
        float[] localX = context.allocateFloatLocalArray(1024);

        // Load input value and compute square
        if (gid < size) {
            localX[lid] = x.get(gid);
            localX[lid] = localX[lid] * localX[lid];
        } else {
            localX[lid] = 0.0f;
        }

        // Perform parallel reduction within the work group
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        // Only the first thread in the work group computes the normalization factor
        if (lid == 0) {
            float ss = localX[0];
            ss /= size;
            ss += ermsNorm;
            ss = 1.0f / TornadoMath.sqrt(ss);
            output.set(0, ss);
        }
    }

    public static void reductionOneBlock2WithLayer(KernelContext context, FloatArray output, FloatArray x,
            FloatArray weights, FloatArray temp,
            IntArray positionAndLayer, int size) {
        int gid = context.globalIdx;

        if (gid < size) {
            // Get the layer offset from positionAndLayer
            int layerOffset = positionAndLayer.get(1) * size;

            // Apply normalization with the correct weight for this layer
            float ss = temp.get(0);
            output.set(gid, weights.get(layerOffset + gid) * (ss * x.get(gid)));
        }
    }

    public static void reductionOneBlock2WithL(KernelContext context, FloatArray output,
            FloatArray weights, FloatArray temp,
            IntArray positionAndLayer, int size) {
        int gid = context.globalIdx;

        if (gid < size) {
            // Get the layer offset from positionAndLayer
            int layerOffset = 0 * size;

            // Apply normalization with the correct weight for this layer
            float ss = temp.get(0);
            output.set(gid, weights.get(layerOffset + gid) * (ss * output.get(gid)));
        }
    }


    public static void mapContextLogits(KernelContext context, FloatArray output,
            FloatArray weights, FloatArray tempLogits, int size) {
        int gid = context.globalIdx;

        if (gid < size) {
            // Apply normalization with weights (no layer offset needed)
            float ss = tempLogits.get(0);
            output.set(gid, weights.get(gid) * (ss * output.get(gid)));
        }
    }




//                    .task("reduce", TornadoVMCompute::reduce, ss, state.wrapX)
//                .task("singleNorm", TornadoVMCompute::singleNorm, ss, config.dim, config.rmsNormEps)
//                .task("mapWithScale", TornadoVMCompute::mapWithScaleAndNorm, state.wrapXb, weights.rms_att_weightFlat,
//            state.wrapX, ss, state.positionAndLayer, config.dim)


    public static void reduce(@Reduce FloatArray output, FloatArray x) {
        output.set(0, 0.0f);
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            float val = x.get(i) * x.get(i);
            output.set(0, output.get(0) + val);
        }
    }


    public static void singleNorm(FloatArray output, int size, float ermsNorm) {
        float ss = output.get(0);
        ss /= size;
        ss += ermsNorm;
        ss = 1.0f / TornadoMath.sqrt(ss);
        output.set(0, ss);
    }

//    public static void mapWithScaleAndNorm(FloatArray output, FloatArray weights, FloatArray input, FloatArray normFactor) {
//        float scale = normFactor.get(0);  // Get the scale computed by singleNorm
//        for (@Parallel int i = 0; i < input.getSize(); i++) {
//            // Now we apply both the scale and the weights
//            output.set(i, weights.get(i) * (scale * input.get(i)));
//        }
//    }

    public static void mapWithScaleAndNorm(FloatArray output, FloatArray weights, FloatArray input,
            FloatArray normFactor, IntArray positionAndLayer, int size) {
        float scale = normFactor.get(0);  // Get the scale computed by singleNorm
        int layerOffset = positionAndLayer.get(1) * size;

        for (@Parallel int i = 0; i < input.getSize(); i++) {
            // Now we apply both the scale and the weights, accounting for layer offset
            output.set(i, weights.get(layerOffset + i) * (scale * input.get(i)));
        }
    }

    public static void rmsnormInnOut(FloatArray output, FloatArray weights,
            int size, float ermsNorm) {
        // Calculate layer offset - weights for this layer start at this offset
        int layerOffset = 0 * size;

        // Calculate sum of squares
        float sumSquares = 0.0f;
        for (int j = 0; j < size; j++) {
            sumSquares += output.get(j) * output.get(j);
        }
        sumSquares /= size;
        sumSquares += ermsNorm; // Add epsilon for numerical stability
        float scale = 1.0f / (float)TornadoMath.sqrt(sumSquares);

        // Normalize and scale with weights from the correct layer
        for (int j = 0; j < size; j++) {
            output.set(j,weights.get(layerOffset + j) * (scale * output.get(j)));
        }
    }

    /**
     * In-place addition using KernelContext
     */
    public static void addInPlace(FloatArray output, FloatArray input) {
        for (@Parallel int i = 0; i < input.getSize(); i++) {
            output.set(i, input.get(i) + output.get(i));
        }
    }

    /**
     * Matrix-vector multiplication for transformer attention computation
     *
     * @param x
     *         Input vector (corresponds to xb in CUDA code)
     * @param xout
     *         Output vector (corresponds to q, k, or v in CUDA code)
     * @param w
     *         Weight matrix (flattened, containing all layers)
     * @param n
     *         Input dimension
     * @param d
     *         Output dimension
     * @param positionAndLayer
     *         Combined position and layer information for weight offset calculation
     */
    public static void matmul(FloatArray xout, FloatArray x, FloatArray w, int n, int d, IntArray positionAndLayer) {
        int layer = positionAndLayer.get(1);
        int layerOffset = layer * n * d;  // Correctly calculates offset based on dimensions


        for (@Parallel int i = 0; i < d; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += w.get(layerOffset + i * n + j) * x.get(j);
            }
            xout.set(i, sum);
        }
    }


    public static void matmulKV(FloatArray xout, FloatArray x, FloatArray w, int dim, int kvdim, IntArray positionAndLayer) {
        int layer = positionAndLayer.get(1);
        int layerOffset = layer * dim * kvdim;

        // Loop over each output element (kvdim outputs)
        for (@Parallel int i = 0; i < kvdim; i++) {
            float sum = 0.0f;
            // Multiply with each input element (dim inputs)
            for (int j = 0; j < dim; j++) {
                // w is organized as [layer, dim, kvdim]
                // For column i, row j, the index is layerOffset + j*dim + i
                sum += w.get(layerOffset + j * dim + i) * x.get(j);
            }
            xout.set(i, sum);
        }
    }


    /**
     * SiLU activation function
     */


    private static void reductionOneBlock(KernelContext context, FloatArray output, FloatArray x) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupSize = context.localGroupSizeX;
        float[] localX = context.allocateFloatLocalArray(1024);
        localX[lid] = x.get(gid);
        localX[lid] = localX[lid] * localX[lid];
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        if (lid == 0) {
            float ss = localX[0];
            ss /= x.getSize();
            ss += 1e-5f;
            ss = 1.0f / TornadoMath.sqrt(ss);
            output.set(0, ss);
        }
    }

    public static void rmsNorm_Step1(KernelContext context, FloatArray partialSums, FloatArray input) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupSize = context.localGroupSizeX;
        int totalSize = input.getSize();

        // Allocate local memory for reduction
        float[] localSums = context.allocateFloatLocalArray(256);

        // Each thread calculates sum of squares for its elements
        float threadSum = 0.0f;
        for (int i = gid; i < totalSize; i += input.getSize()) {
            float val = input.get(i);
            threadSum += val * val;
        }

        // Load thread's sum into local memory
        localSums[lid] = threadSum;

        // Synchronize
        context.localBarrier();

        // Perform reduction in local memory
        for (int stride = groupSize / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                localSums[lid] += localSums[lid + stride];
            }
            context.localBarrier();
        }

        // First thread in work group writes the partial sum
        if (lid == 0) {
            partialSums.set(context.globalGroupSizeX, localSums[0]);
        }
    }

    public static void reduceSquareSums(KernelContext context, FloatArray input, FloatArray reduce, int totalSize) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;
        int localGroupSize = context.localGroupSizeX;
        int groupID = context.groupIdx;

        // Allocate local memory
        float[] localSum = context.allocateFloatLocalArray(128);

        // Calculate thread's sum of squares
        float threadSum = 0.0f;
        for (int i = globalIdx; i < totalSize; i += totalSize) {
            float val = input.get(i);
            threadSum += val * val;
        }

        // Store in local memory
        localSum[localIdx] = threadSum;

        // Synchronize
        context.localBarrier();

        // Perform reduction in local memory
        for (int stride = localGroupSize / 2; stride > 0; stride >>= 1) {
            if (localIdx < stride) {
                localSum[localIdx] += localSum[localIdx + stride];
            }
            context.localBarrier();
        }

        // Only the first thread in each work group writes the result
        if (localIdx == 0) {
            reduce.set(groupID, localSum[0]);
        }
    }

    public static void finalSum(KernelContext context, FloatArray scaleFactor, FloatArray reduce, int totalSize, float epsilon) {
        // Only execute in the first thread
        if (context.globalIdx == 0) {
            float sumSquares = 0.0f;

            // Sum all partial results
            for (int i = 0; i < reduce.getSize(); i++) {
                sumSquares += reduce.get(i);
            }

            // Calculate RMS norm scaling factor exactly like the sequential version
            sumSquares /= totalSize;
            sumSquares += epsilon;
            float scale = 1.0f / TornadoMath.sqrt(sumSquares);

            // Store the result
            scaleFactor.set(0, scale);
        }
    }

    public static void matmulHybrid(KernelContext context, FloatArray xout, FloatArray x, FloatArray w,
            int n, int d, IntArray positionAndLayer) {
        int globalIdx = context.globalIdx;

        if (globalIdx < d) {
            int layer = positionAndLayer.get(1);
            int layerOffset = layer * n * d;
            int baseIdx = layerOffset + globalIdx * n;

            // Use a fixed local memory size for frequently accessed items
            // Use much smaller tile size to minimize boundary issues
            final int CHUNK = 32;
            float sum = 0.0f;

            for (int start = 0; start < n; start += CHUNK) {
                int end = TornadoMath.min(start + CHUNK, n);

                // Process this chunk directly, no local memory for simplicity
                for (int j = start; j < end; j += 4) {
                    float sum1 = (j < n) ? w.get(baseIdx + j) * x.get(j) : 0;
                    float sum2 = (j+1 < n) ? w.get(baseIdx + j+1) * x.get(j+1) : 0;
                    float sum3 = (j+2 < n) ? w.get(baseIdx + j+2) * x.get(j+2) : 0;
                    float sum4 = (j+3 < n) ? w.get(baseIdx + j+3) * x.get(j+3) : 0;
                    sum += sum1 + sum2 + sum3 + sum4;
                }
            }

            xout.set(globalIdx, sum);
        }
    }

//    [1 1 1
//     1  1  1]  X  [1 1 1]

    public static void matmulUnroll4(FloatArray xout, FloatArray x, FloatArray w, int n, int d, IntArray positionAndLayer) {
        int layer = positionAndLayer.get(1);
        int layerOffset = layer * n * d;

        // Simple mapping to global threads, assuming hardware handles work distribution
        for (@Parallel int i = 0; i < d; i++) {
            float sum = 0.0f;
            int baseIdx = layerOffset + i * n;

            // For very large n, consider chunking this loop
            for (int j = 0; j < n; j += 4) {
                // Unrolled to process 4 elements at once (adjust based on your vector width)
                float sum1 = (j < n) ? w.get(baseIdx + j) * x.get(j) : 0;
                float sum2 = (j+1 < n) ? w.get(baseIdx + j+1) * x.get(j+1) : 0;
                float sum3 = (j+2 < n) ? w.get(baseIdx + j+2) * x.get(j+2) : 0;
                float sum4 = (j+3 < n) ? w.get(baseIdx + j+3) * x.get(j+3) : 0;

                sum += sum1 + sum2 + sum3 + sum4;
            }

            xout.set(i, sum);
        }
    }

    public static void matmulUnroll4WithResidual(FloatArray xout, FloatArray x, FloatArray w,
            int n, int d, IntArray positionAndLayer) {
        int layer = positionAndLayer.get(1);
        int layerOffset = layer * n * d;

        // Simple mapping to global threads, assuming hardware handles work distribution
        for (@Parallel int i = 0; i < d; i++) { //<--- projectionTwo d = 2048
            float sum = 0.0f;
            int baseIdx = layerOffset + i * n;

            // For very large n, consider chunking this loop
            for (int j = 0; j < n; j += 4) { // <--- projectionTwo n = 8192
                // Unrolled to process 4 elements at once
                float sum1 = (j < n) ? w.get(baseIdx + j) * x.get(j) : 0;
                float sum2 = (j+1 < n) ? w.get(baseIdx + j+1) * x.get(j+1) : 0;
                float sum3 = (j+2 < n) ? w.get(baseIdx + j+2) * x.get(j+2) : 0;
                float sum4 = (j+3 < n) ? w.get(baseIdx + j+3) * x.get(j+3) : 0;

                sum += sum1 + sum2 + sum3 + sum4;
            }

            // Add to existing output
            xout.set(i, sum + xout.get(i));
        }
    }

    public static void matmulUnroll4WithResidual(FloatArray xout, FloatArray x, VectorFloat4 w,
            int n, int d, IntArray positionAndLayer) {
        int layer = positionAndLayer.get(1);
        int layerOffset = layer * n * d / 4; // Divide by 4 since we're using VectorFloat4

        for (@Parallel int i = 0; i < d; i++) {
            float sum = 0.0f;
            int baseIdx = layerOffset + i * n / 4; // Divide by 4 since each VectorFloat4 contains 4 values

            for (int j = 0; j < n / 4; j++) {
                // Get 4 input values at once
                float x0 = x.get(j * 4);
                float x1 = x.get(j * 4 + 1);
                float x2 = x.get(j * 4 + 2);
                float x3 = x.get(j * 4 + 3);

                // Get 4 weight values at once
                Float4 weights = w.get(baseIdx + j);

                // Process 4 elements at once using vector operations
                sum += weights.getX() * x0 + weights.getY() * x1 + weights.getZ() * x2 + weights.getW() * x3;
            }

            // Add to existing output (residual connection)
            xout.set(i, sum + xout.get(i));
        }
    }

    public static void matmulUnroll4WithResidualX(FloatArray xout, FloatArray x, VectorFloat4 w,
            int n, int d, IntArray positionAndLayer) {
        int layer = positionAndLayer.get(1);
        int layerOffset = layer * n * d / 4; // Divide by 4 since we're using VectorFloat4

        for (@Parallel int i = 0; i < d; i++) {
            float sum = 0.0f;
            int baseIdx = layerOffset + i * n / 4; // Divide by 4 since each VectorFloat4 contains 4 values

            for (int j = 0; j < n / 4; j++) {
                // Get 4 input values at once
                Float4 xLoad = new Float4(x.get(j * 4), x.get(j * 4 + 1), x.get(j * 4 + 2), x.get(j * 4 + 3));
                // Get 4 weight values at once
                Float4 weights = w.get(baseIdx + j);

                Float4.dot(weights, xLoad);
                sum += Float4.dot(weights, xLoad);
            }

            // Add to existing output (residual connection)
            xout.set(i, sum + xout.get(i));
        }
    }


    public static void siluElemWiseMulActivation(int hidenDimSize, FloatArray hb, FloatArray hb2) {
        for (@Parallel int i = 0; i < hidenDimSize; i++) {
            float val = hb.get(i);
            val *= (1.0f / (1.0f + TornadoMath.exp(-val)));
            val *= hb2.get(i);
            hb.set(i, val);
        }
    }



    public static void  combinedMatmulSiluActivation(KernelContext context, FloatArray hb, FloatArray x, FloatArray w,
            int n, int d, IntArray positionAndLayer) {
        // Get thread ID
        int i = context.globalIdx;


        int layer = positionAndLayer.get(1);
        int layerOffset = layer * n * d;
        int baseIdx = layerOffset + i * n;

        // Matrix multiplication - without tiling for now to ensure correctness
        float sum = 0.0f;
        for (int j = 0; j < n; j += 4) {
            // Process 4 elements at a time (unrolled)
            float sum1 = (j < n) ? w.get(baseIdx + j) * x.get(j) : 0;
            float sum2 = (j+1 < n) ? w.get(baseIdx + j+1) * x.get(j+1) : 0;
            float sum3 = (j+2 < n) ? w.get(baseIdx + j+2) * x.get(j+2) : 0;
            float sum4 = (j+3 < n) ? w.get(baseIdx + j+3) * x.get(j+3) : 0;

            sum += sum1 + sum2 + sum3 + sum4;
        }

        // SiLU activation and element-wise multiplication
        float hbVal = hb.get(i);
        float silu = hbVal * (1.0f / (1.0f + TornadoMath.exp(-hbVal)));
        float result = silu * sum;

        // Store final result
        hb.set(i, result);
    }

//    public static void siluElemWiseMulActivation(int hidenDimSize, FloatArray hb, FloatArray hb2) {
//        for (@Parallel int i = 0; i < hidenDimSize; i++) {
//            float val = hb.get(i);
//            val *= (1.0f / (1.0f + TornadoMath.exp(-val)));
//            val *= hb2.get(i);
//            hb.set(i, val);
//        }
//    }

    public static void combinedMatmulSiluActivation(FloatArray hb, FloatArray x, FloatArray w,
            int n, int d, IntArray positionAndLayer) {
        int layer = positionAndLayer.get(1);
        int layerOffset = layer * n * d;

        // Parallel processing for each output element
        for (@Parallel int i = 0; i < d; i++) {
            // Part 1: Matrix multiplication (equivalent to matmulUnroll4)
            float sum = 0.0f;
            int baseIdx = layerOffset + i * n;

            for (int j = 0; j < n; j += 4) {
                float sum1 = (j < n) ? w.get(baseIdx + j) * x.get(j) : 0;
                float sum2 = (j+1 < n) ? w.get(baseIdx + j+1) * x.get(j+1) : 0;
                float sum3 = (j+2 < n) ? w.get(baseIdx + j+2) * x.get(j+2) : 0;
                float sum4 = (j+3 < n) ? w.get(baseIdx + j+3) * x.get(j+3) : 0;

                sum += sum1 + sum2 + sum3 + sum4;
            }

            // Part 2: SiLU activation and element-wise multiplication
            // In the original code:
            // 1. Matrix multiplication results are stored in hb2
            // 2. hb undergoes SiLU activation and is multiplied by hb2

            // Calculate SiLU on the hb input value: x * sigmoid(x)
            float hbVal = hb.get(i);
            float silu = hbVal * (1.0f / (1.0f + TornadoMath.exp(-hbVal)));

            // Multiply SiLU result by the matrix multiplication result (sum)
            float result = silu * sum;

            // Store the final result back in hb
            hb.set(i, result);
        }
    }

    public static void ffnMatvecSiluFused(FloatArray hb, FloatArray x,
            FloatArray gate_w, FloatArray up_w,
            IntArray positionAndLayer, int dim, int hiddenDim) {
        int layer = positionAndLayer.get(1);
        int gateOffset = layer * dim * hiddenDim;
        int upOffset = layer * dim * hiddenDim;

        // Calculate gate and up projections with SiLU in one fused kernel
        for (@Parallel int i = 0; i < hiddenDim; i++) {
            float gateSum = 0.0f, upSum = 0.0f;

            // Unrolled loops for better performance
            for (int j = 0; j < dim; j += 4) {
                // Gate projection with unrolling
                gateSum += (j < dim) ? gate_w.get(gateOffset + i * dim + j) * x.get(j) : 0;
                gateSum += (j+1 < dim) ? gate_w.get(gateOffset + i * dim + j+1) * x.get(j+1) : 0;
                gateSum += (j+2 < dim) ? gate_w.get(gateOffset + i * dim + j+2) * x.get(j+2) : 0;
                gateSum += (j+3 < dim) ? gate_w.get(gateOffset + i * dim + j+3) * x.get(j+3) : 0;

                // Up projection with unrolling
                upSum += (j < dim) ? up_w.get(upOffset + i * dim + j) * x.get(j) : 0;
                upSum += (j+1 < dim) ? up_w.get(upOffset + i * dim + j+1) * x.get(j+1) : 0;
                upSum += (j+2 < dim) ? up_w.get(upOffset + i * dim + j+2) * x.get(j+2) : 0;
                upSum += (j+3 < dim) ? up_w.get(upOffset + i * dim + j+3) * x.get(j+3) : 0;
            }

            // Apply SiLU activation to gate and multiply with up projection in one step
            float silu = gateSum * (1.0f / (1.0f + TornadoMath.exp(-gateSum)));
            hb.set(i, silu * upSum);
        }
    }

    /**
     * Optimized matrix multiplication kernel using tiling and local memory
     * techniques observed in the TornadoVM framework examples.
     *
     * @param xout Output matrix array
     * @param x Input vector
     * @param w Weight matrix
     * @param n Size of input dimension
     * @param d Size of output dimension
     * @param positionAndLayer Layer position information
     */
    public static void matmulOptimized(KernelContext context, FloatArray xout, FloatArray x, FloatArray w,
            int n, int d, IntArray positionAndLayer) {
        // Get thread identification from KernelContext
        int globalIdx = context.globalIdx;

        // Process only if thread ID is within bounds
        if (globalIdx < d) {
            int layer = positionAndLayer.get(7); // Based on the OpenCL kernel
            int layerOffset = layer << 24; // Equivalent to layer * 2^24

            // Base index calculation for weights
            int baseIdx = layerOffset + (globalIdx << 11); // Equivalent to globalIdx * 2048

            // Allocate local memory for input vector (helps with memory coalescing)
            final int TILE_SIZE = 128; // Adjust based on hardware characteristics
            float[] localX = context.allocateFloatLocalArray(TILE_SIZE);

            float sum = 0.0f;

            // Process input in tiles to improve cache utilization
            for (int tileStart = 0; tileStart < n; tileStart += TILE_SIZE) {
                // Load a tile of input vector into local memory
                int localIdx = context.localIdx;
                if (localIdx < TILE_SIZE && tileStart + localIdx < n) {
                    localX[localIdx] = x.get(tileStart + localIdx);
                }

                // Wait for all threads to load data
                context.localBarrier();

                // Process the current tile with unrolling
                int tileEnd = Math.min(tileStart + TILE_SIZE, n);
                for (int j = tileStart; j < tileEnd; j += 4) {
                    // Unrolled multiplication of 4 elements (or fewer if at the end)
                    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;

                    // Use local memory for input vector access
                    int localOffset = j - tileStart;

                    if (j < tileEnd)
                        sum1 = w.get(baseIdx + j) * localX[localOffset];
                    if (j+1 < tileEnd)
                        sum2 = w.get(baseIdx + j+1) * localX[localOffset+1];
                    if (j+2 < tileEnd)
                        sum3 = w.get(baseIdx + j+2) * localX[localOffset+2];
                    if (j+3 < tileEnd)
                        sum4 = w.get(baseIdx + j+3) * localX[localOffset+3];

                    sum += sum1 + sum2 + sum3 + sum4;
                }

                // Barrier to ensure all threads are done with this tile before loading next
                context.localBarrier();
            }

            // Write result to output
            xout.set(globalIdx, sum);
        }
    }

    public static void matmulCollaborative(KernelContext context, FloatArray xout, FloatArray x, FloatArray w,
            int n, int d, IntArray positionAndLayer) {
        // Get thread identification
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;
        int groupIdx = context.groupIdx;
        int localSize = context.localGroupSizeX;

        // Calculate indices
        int layer = positionAndLayer.get(1);
        int layerOffset = layer * n * d;
        int baseIdx = layerOffset + globalIdx * n;

        // Define tile size for local memory
        final int TS = 128; // Can be tuned for your hardware

        // Allocate local memory for input vector tile
        float[] xLocal = context.allocateFloatLocalArray(TS);

        // Initialize accumulator
        float sum = 0.0f;

        // Process input in tiles
        for (int tileStart = 0; tileStart < n; tileStart += TS) {
            int tileEnd = uk.ac.manchester.tornado.api.math.TornadoMath.min(tileStart + TS, n);
            int tileSize = tileEnd - tileStart;

            // Collaborative loading of input vector into local memory
            for (int i = localIdx; i < tileSize; i += localSize) {
                xLocal[i] = x.get(tileStart + i);
            }

            // Wait for all threads to finish loading
            context.localBarrier();

            // Process the current tile with unrolling
            for (int j = 0; j < tileSize; j += 4) {
                // Bounds-checked unrolled computation
                float sum1 = (j < tileSize) ? w.get(baseIdx + tileStart + j) * xLocal[j] : 0.0f;
                float sum2 = (j+1 < tileSize) ? w.get(baseIdx + tileStart + j+1) * xLocal[j+1] : 0.0f;
                float sum3 = (j+2 < tileSize) ? w.get(baseIdx + tileStart + j+2) * xLocal[j+2] : 0.0f;
                float sum4 = (j+3 < tileSize) ? w.get(baseIdx + tileStart + j+3) * xLocal[j+3] : 0.0f;

                sum += sum1 + sum2 + sum3 + sum4;
            }

            // Wait for all threads to finish with this tile
            context.localBarrier();
        }

        // Write result
        if (globalIdx < d) {
            xout.set(globalIdx, sum);
        }
    }


    /**
     * Optimized implementation with work-group collaboration
     * This version uses collaborative loading to improve memory access patterns
     */
    public static void matmulCollaborativXe(KernelContext context, FloatArray xout,
            FloatArray x, FloatArray w,
            int n, int d, IntArray positionAndLayer) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;
        int localSize = context.localGroupSizeX;
        int groupId = context.groupIdx;

        // Exit if thread is out of bounds
//        if (globalIdx >= d) return;

        int layer = positionAndLayer.get(1);
        int layerOffset = layer << 24;
        int baseIdx = layerOffset + (globalIdx << 11);

        // Use shared memory for input vector chunks
        final int CHUNK_SIZE = 128; // Tune based on hardware
        float[] localX = context.allocateFloatLocalArray(CHUNK_SIZE);
        float sum = 0.0f;

        // Process in chunks to maximize cache efficiency
        for (int chunkStart = 0; chunkStart < n; chunkStart += CHUNK_SIZE) {
            // Collaborative loading of input chunk
            for (int i = localIdx; i < CHUNK_SIZE && chunkStart + i < n; i += localSize) {
                localX[i] = x.get(chunkStart + i);
            }

            // Wait for all threads to finish loading
            context.localBarrier();

            // Process chunk with unrolling (4 elements at a time)
            int chunkEnd = TornadoMath.min(chunkStart + CHUNK_SIZE, n);
            for (int j = chunkStart; j < chunkEnd; j += 4) {
                // Vector-style processing with local memory
                for (int k = 0; k < 4 && j + k < chunkEnd; k++) {
                    int localIndex = j + k - chunkStart;
                    sum += w.get(baseIdx + j + k) * localX[localIndex];
                }
            }

            // Synchronize before next chunk
            context.localBarrier();
        }

        // Store result
        xout.set(globalIdx, sum);
    }

    public static void matmulUnroll8(FloatArray xout, FloatArray x, FloatArray w, int n, int d, IntArray positionAndLayer) {
        int layer = positionAndLayer.get(1);
        int layerOffset = layer * n * d;

        // Simple mapping to global threads, assuming hardware handles work distribution
        for (@Parallel int i = 0; i < d; i++) {
            float sum = 0.0f;
            int baseIdx = layerOffset + i * n;

            // For very large n, consider chunking this loop
            for (int j = 0; j < n; j += 8) {
                // Unrolled to process 4 elements at once (adjust based on your vector width)
                float sum1 = (j < n) ? w.get(baseIdx + j) * x.get(j) : 0;
                float sum2 = (j+1 < n) ? w.get(baseIdx + j+1) * x.get(j+1) : 0;
                float sum3 = (j+2 < n) ? w.get(baseIdx + j+2) * x.get(j+2) : 0;
                float sum4 = (j+3 < n) ? w.get(baseIdx + j+3) * x.get(j+3) : 0;
                float sum5 = (j+4 < n) ? w.get(baseIdx + j+4) * x.get(j+4) : 0;
                float sum6 = (j+5 < n) ? w.get(baseIdx + j+5) * x.get(j+5) : 0;
                float sum7 = (j+6 < n) ? w.get(baseIdx + j+6) * x.get(j+6) : 0;
                float sum8 = (j+7 < n) ? w.get(baseIdx + j+7) * x.get(j+7) : 0;

                sum += sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8;
            }

            xout.set(i, sum);
        }
    }

    public static void matmulUnroll16(FloatArray xout, FloatArray x, FloatArray w, int n, int d, IntArray positionAndLayer) {
        int layer = positionAndLayer.get(1);
        int layerOffset = layer * n * d;

        // Simple mapping to global threads, assuming hardware handles work distribution
        for (@Parallel int i = 0; i < d; i++) {
            float sum = 0.0f;
            int baseIdx = layerOffset + i * n;

            // For very large n, consider chunking this loop
            for (int j = 0; j < n; j += 16) {
                // Unrolled to process 16 elements at once
                float sum1 = (j < n) ? w.get(baseIdx + j) * x.get(j) : 0;
                float sum2 = (j+1 < n) ? w.get(baseIdx + j+1) * x.get(j+1) : 0;
                float sum3 = (j+2 < n) ? w.get(baseIdx + j+2) * x.get(j+2) : 0;
                float sum4 = (j+3 < n) ? w.get(baseIdx + j+3) * x.get(j+3) : 0;
                float sum5 = (j+4 < n) ? w.get(baseIdx + j+4) * x.get(j+4) : 0;
                float sum6 = (j+5 < n) ? w.get(baseIdx + j+5) * x.get(j+5) : 0;
                float sum7 = (j+6 < n) ? w.get(baseIdx + j+6) * x.get(j+6) : 0;
                float sum8 = (j+7 < n) ? w.get(baseIdx + j+7) * x.get(j+7) : 0;
                float sum9 = (j+8 < n) ? w.get(baseIdx + j+8) * x.get(j+8) : 0;
                float sum10 = (j+9 < n) ? w.get(baseIdx + j+9) * x.get(j+9) : 0;
                float sum11 = (j+10 < n) ? w.get(baseIdx + j+10) * x.get(j+10) : 0;
                float sum12 = (j+11 < n) ? w.get(baseIdx + j+11) * x.get(j+11) : 0;
                float sum13 = (j+12 < n) ? w.get(baseIdx + j+12) * x.get(j+12) : 0;
                float sum14 = (j+13 < n) ? w.get(baseIdx + j+13) * x.get(j+13) : 0;
                float sum15 = (j+14 < n) ? w.get(baseIdx + j+14) * x.get(j+14) : 0;
                float sum16 = (j+15 < n) ? w.get(baseIdx + j+15) * x.get(j+15) : 0;

                sum += sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8 +
                        sum9 + sum10 + sum11 + sum12 + sum13 + sum14 + sum15 + sum16;
            }

            xout.set(i, sum);
        }
    }

    public static void normalizeAndScale(KernelContext context, FloatArray output, FloatArray input, FloatArray weights, FloatArray scaleFactor, IntArray positionAndLayer, int size) {
        int globalIdx = context.globalIdx;

        // Only process if within bounds
        if (globalIdx < size) {
            // Get the scaling factor (same for all threads)
            float scale = scaleFactor.get(0);

            // Calculate layer offset exactly like the sequential version
            int layerOffset = positionAndLayer.get(1) * size;

            // Normalize and scale with weights exactly like the sequential version
            output.set(globalIdx, weights.get(layerOffset + globalIdx) * (scale * input.get(globalIdx)));
        }
    }

    public static void rmsNorm_Step2(KernelContext context, FloatArray scaleFactor, FloatArray partialSums, int totalSize, float epsilon) {
        // Only needs to be executed by a single thread
            // Combine all partial sums
        float sumSquares = 0.0f;
        for (int i = 0; i < partialSums.getSize(); i++) {
            sumSquares += partialSums.get(i);
        }

        // Calculate RMS norm factor
        sumSquares /= totalSize;
        sumSquares += epsilon;
        float scale = 1.0f / TornadoMath.sqrt(sumSquares);

        // Store the scaling factor
        scaleFactor.set(0, scale);
    }

    public static void rmsNorm_Step3(KernelContext context, FloatArray output, FloatArray input, FloatArray weights, FloatArray scaleFactor, IntArray positionAndLayer, int size) {
        int gid = context.globalIdx;

        // Only process if within bounds

        float scale = scaleFactor.get(0);
        int layerOffset = positionAndLayer.get(1) * size;

        // Apply normalization and weights
        output.set(gid, weights.get(layerOffset + gid) * (scale * input.get(gid)));
    }

//    private static void reductionOneBlock2(KernelContext context, FloatArray output, FloatArray x, FloatArray weights, FloatArray temp) {
//        int gid = context.globalIdx;
//        float ss = temp.get(0);
//        output.set(gid, weights.get(gid) * (ss * x.get(gid)));
//    }

    public static void reductionOneBlock(KernelContext context, FloatArray output, FloatArray x, int localSize, float rmsNormEps) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory for reduction
        float[] localX = context.allocateFloatLocalArray(localSize);

        // Initialize local memory
        if (lid < localSize) {
            localX[lid] = 0.0f;
        }

        // Load data into local memory with stride to cover all elements
        float sum = 0.0f;
        for (int i = gid; i < x.getSize(); i += x.getSize()) {
            float val = x.get(i);
            sum += val * val;
        }
        localX[lid] = sum;

        // Synchronize to make sure all loads are done
        context.localBarrier();

        // Perform reduction in local memory
        for (int stride = groupSize / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
            context.localBarrier();
        }

        // Write the result
        if (lid == 0) {
            float ss = localX[0];
            ss /= x.getSize(); // Normalize by full size
            ss += rmsNormEps;
            ss = 1.0f / TornadoMath.sqrt(ss);
            output.set(0, ss);
        }
    }

    public static void reductionOneBlock2(KernelContext context, FloatArray output, FloatArray x, FloatArray weights, FloatArray temp, IntArray positioNlayer, int size) {
        int gid = context.globalIdx;
        float ss = temp.get(0);
        int layerOffset = positioNlayer.get(1) * size;

        output.set(gid, weights.get(layerOffset + gid) * (ss * x.get(gid)));
    }

    public static void reductionOneBlock2InNout(KernelContext context, FloatArray x, FloatArray weights, FloatArray temp, IntArray positioNlayer, int size) {
        int gid = context.globalIdx;
        float ss = temp.get(0);
        int layerOffset = positioNlayer.get(1) * size;

        x.set(gid, weights.get(layerOffset + gid) * (ss * x.get(gid)));
    }

    public static void emptyTaskToForceCopyIn(FloatArray buffer) {
        float dummy = buffer.get(0);
        if (dummy > Float.MAX_VALUE) {
            buffer.set(0, dummy);
        }
    }

    public static void matmulTornadoQ4Pure(ByteArray thisx, FloatArray that, FloatArray out, int dim1, int vocabSize) {
        final int BLOCK_SIZE = GGMLType.Q4_0.getBlockSize(); // Q4 block size
        final int BYTES_PER_BLOCK = GGMLType.Q4_0.getTypeSize(); // The block size in bytes for Q4

        //        int idx = context.globalIdx;

        for (@Parallel int idx = 0; idx < vocabSize; idx++) {
            float result = 0f;
            int thisOffset = idx * dim1;

            for (int j = 0; j < dim1; j++) {
                int index = thisOffset + j;

                // Calculate block position and within-block index
                int blockIndex = index / BLOCK_SIZE;
                int withinBlockIndex = index % BLOCK_SIZE;
                int blockOffset = blockIndex * BYTES_PER_BLOCK;

                // Decode quantized value and scale
                float dequantizedValue = decodeQ4(thisx, blockOffset, withinBlockIndex);

                // Multiply the dequantized value by the corresponding element in 'that'
                result += dequantizedValue * that.get(j);
            }

            // Store the result in the output array
            out.set(idx, result);
        }
    }


    public static void forcePropagationTwoArrays(FloatArray x, FloatArray y) {
        x.set(0, x.get(0));
        y.set(0, y.get(0));
    }


    public static void ropeRotation(KernelContext context, IntArray positionNlayer, FloatArray sq, FloatArray sk, int kv_dim, int head_size) {
        int i = context.globalIdx * 2;

        int head_dim = i % head_size;
        // 50000.0f vs 10000.0f
        float freq = 1.0f / TornadoMath.pow(50000.0f, head_dim / (float) head_size);
        float val = positionNlayer.get(0) * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only

        // Rotate query vector
        float v0q = sq.get(i);
        float v1q = sq.get(i + 1);
        sq.set(i, v0q * fcr - v1q * fci);
        sq.set(i + 1, v0q * fci + v1q * fcr);

        // Rotate key vector if needed
        if (rotn > 1 && i < sk.getSize()) {
            float v0k = sk.get(i);
            float v1k = sk.get(i + 1);
            sk.set(i, v0k * fcr - v1k * fci);
            sk.set(i + 1, v0k * fci + v1k * fcr);
        }

    }

    public static void ropeRotationSerial(IntArray position, FloatArray sq, FloatArray sk,
            int n_heads, int n_kv_heads, int head_size,
            FloatArray freq_cis_real, FloatArray freq_cis_imag)
    {
        int pos = position.get(0);

        // Loop over all heads
        for (int i = 0; i < n_heads; i++) {
            // Loop over dimensions within each head
            for (int j = 0; j < head_size; j += 2) {
                // Get precomputed rotation values for this position and dimension
                int head_dim = j / 2;  // Since we're incrementing by 2
                int freq_index = pos * (head_size / 2) + head_dim;

                float fcr = freq_cis_real.get(freq_index);
                float fci = freq_cis_imag.get(freq_index);

                // Calculate indices for the current head and dimension
                int qIdx = i * head_size + j;

                // Rotate query vector
                float q0 = sq.get(qIdx);
                float q1 = sq.get(qIdx + 1);
                sq.set(qIdx, q0 * fcr - q1 * fci);
                sq.set(qIdx + 1, q0 * fci + q1 * fcr);

                // Rotate key vector if this is a KV head
                if (i < n_kv_heads) {
                    int kIdx = i * head_size + j;
                    float k0 = sk.get(kIdx);
                    float k1 = sk.get(kIdx + 1);
                    sk.set(kIdx, k0 * fcr - k1 * fci);
                    sk.set(kIdx + 1, k0 * fci + k1 * fcr);
                }
            }
        }
    }


    public static void ropeRotationSerialX(IntArray position, FloatArray sq, FloatArray sk, int kv_dim, int head_size) {
        // Process each pair of adjacent values
        for (int i = 0; i < sq.getSize(); i += 2) {
            // Calculate which feature dimension we're working with
            int head_dim = i % head_size;

            // Calculate frequency for this dimension
            float freq = 1.0f / (float) TornadoMath.pow(10000.0f, head_dim / (float) head_size);

            // Calculate rotation angle
            float val = position.get(0) * freq;
            float fcr = (float) TornadoMath.cos(val);
            float fci = (float) TornadoMath.sin(val);

            // Determine if we need to rotate just query or both query and key
            int rotn = i < kv_dim ? 2 : 1;

            // Rotate query vector
            float v0q = sq.get(i);
            float v1q = sq.get(i + 1);
            sq.set(i, v0q * fcr - v1q * fci);
            sq.set(i + 1, v0q * fci + v1q * fcr);

            // Rotate key vector if needed
            if (rotn > 1 && i < sk.getSize()) {
                float v0k = sk.get(i);
                float v1k = sk.get(i + 1);
                sk.set(i, v0k * fcr - v1k * fci);
                sk.set(i + 1, v0k * fci + v1k * fcr);
            }
        }
    }

    public static void matmulTornadoQ4(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
        final int BLOCK_SIZE = GGMLType.Q4_0.getBlockSize(); // Q4 block size
        final int BYTES_PER_BLOCK = GGMLType.Q4_0.getTypeSize(); // Bytes per block for Q4
        final int TS = 16;

        // Thread-local identifiers
        int localRow = context.localIdx;
        int localCol = context.localIdy;

        int globalRow = TS * context.groupIdx + localRow;
        int globalCol = TS * context.groupIdy + localCol;

        // Allocate shared memory (local arrays) for tiles
        float[] sharedThat = context.allocateFloatLocalArray(TS); // Local copy of 'that' for reuse
        float[] sharedThisx = context.allocateFloatLocalArray(TS); // Local copy of quantized 'thisx'

        float sum = 0.0f; // Accumulate the results

        // Loop over all tiles
        int numTiles = dim1 / TS;
        for (int tileIndex = 0; tileIndex < numTiles; tileIndex++) {
            int tiledRow = TS * tileIndex + localRow;
            int tiledCol = TS * tileIndex + localCol;

            // Load one tile of A (thisx) and B (that) into shared memory
            if (tiledRow < dim1 && globalRow < dim1) {
                sharedThisx[localCol] = decodeQ4(thisx, tiledRow / BLOCK_SIZE * BYTES_PER_BLOCK, tiledRow % BLOCK_SIZE);
            } else {
                sharedThisx[localCol] = 0.0f; // Out-of-bound guard
            }

            if (tiledCol < dim1 && globalCol < dim1) {
                sharedThat[localRow] = that.get(tiledCol);
            } else {
                sharedThat[localRow] = 0.0f; // Out-of-bound guard
            }

            // Synchronise to ensure all threads have loaded the tile
            context.localBarrier();

            // Perform computation for the tile
            for (int k = 0; k < TS; k++) {
                sum += sharedThisx[k] * sharedThat[k];
            }

            // Synchronise before loading the next tile
            context.localBarrier();
        }

        // Store the result
        if (globalRow < dim1 && globalCol < dim1) {
            out.set(globalCol * dim1 + globalRow, sum);
        }
    }

    public static void matmulTornadoQ42(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
        final int BLOCK_SIZE = GGMLType.Q4_0.getBlockSize(); // Q4 block size
        final int BYTES_PER_BLOCK = GGMLType.Q4_0.getTypeSize(); // The block size in bytes for Q4

        int idx = context.globalIdx;

        float result = 0f;
        int thisOffset = idx * dim1;

        for (int j = 0; j < dim1; j++) {
            int index = thisOffset + j;

            // Calculate block position and within-block index
            int blockIndex = index / BLOCK_SIZE;
            int withinBlockIndex = index % BLOCK_SIZE;
            int blockOffset = blockIndex * BYTES_PER_BLOCK;

            // Decode quantized value and scale
            float dequantizedValue = decodeQ4(thisx, blockOffset, withinBlockIndex);

            // Multiply the dequantized value by the corresponding element in 'that'
            result += dequantizedValue * that.get(j);
        }

        // Store the result in the output array
        out.set(idx, result);
    }

    private static float pow2(int n) {
        if (n >= 0) {
            if (n < 31) {
                return (float) (1 << n);
            }
            return Float.POSITIVE_INFINITY;
        }
        if (n > -150) {
            return 1.0f / (1 << -n);
        }
        return 0.0f;
    }

    private static float decodeQ4(ByteArray thisx, int blockOffset, int withinBlockIndex) {
        // Read the scale (Float16 format)
        int scaleByte1 = thisx.get(blockOffset) & 0xFF;
        int scaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
        short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
        float scale = decodeFloat16(scaleFloat16);

        // Determine the byte in which the quantized value is stored
        byte quant;
        if (withinBlockIndex < GGMLType.Q4_0.getBlockSize() / 2) {
            // The lower 4 bits of the byte
            quant = (byte) (thisx.get(blockOffset + Float16.BYTES + withinBlockIndex) & 0x0F);
        } else {
            // The higher 4 bits of the byte
            quant = (byte) ((thisx.get(blockOffset + Float16.BYTES + withinBlockIndex - GGMLType.Q4_0.getBlockSize() / 2) >>> 4) & 0x0F);
        }

        // Dequantize by shifting the value to the range [-8, 7]
        quant -= 8;

        // Return the dequantized value
        return quant * scale;
    }

    public static void matmulTornadoQ8(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
        final int BLOCK_SIZE = 32; // Block size used in quantization
        final int BYTES_PER_BLOCK = 2 + BLOCK_SIZE; // 2 bytes for scale + block_size bytes for values
        final int UNROLL_FACTOR = 8; // Unroll the inner loop for better performance

        int idx = context.globalIdx;
        float result = 0f;
        int thisOffset = idx * dim1;

        // Cache last block index and scale to avoid redundant decoding
        int lastBlockIndex = -1;
        float cachedScale = 0f;

        // Main loop with unrolling
        int j = 0;
        for (; j <= dim1 - UNROLL_FACTOR; j += UNROLL_FACTOR) {
            // Process UNROLL_FACTOR elements at once
            for (int k = 0; k < UNROLL_FACTOR; k++) {
                int index = thisOffset + j + k;
                int blockIndex = index / BLOCK_SIZE;
                int withinBlockIndex = index % BLOCK_SIZE;
                int blockOffset = blockIndex * BYTES_PER_BLOCK;

                // Only decode scale if we're in a new block
                float scale;
                if (blockIndex != lastBlockIndex) {
                    int scaleByte1 = thisx.get(blockOffset) & 0xFF;
                    int scaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
                    short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
                    cachedScale = decodeFloat16Fast(scaleFloat16);
                    lastBlockIndex = blockIndex;
                }
                scale = cachedScale;

                // Read quantized value
                byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);

                // Dequantize and accumulate
                result = fma(quantized * scale, that.get(j + k), result);
            }
        }

        // Handle remaining elements
        for (; j < dim1; j++) {
            int index = thisOffset + j;
            int blockIndex = index / BLOCK_SIZE;
            int withinBlockIndex = index % BLOCK_SIZE;
            int blockOffset = blockIndex * BYTES_PER_BLOCK;

            // Only decode scale if we're in a new block
            float scale;
            if (blockIndex != lastBlockIndex) {
                int scaleByte1 = thisx.get(blockOffset) & 0xFF;
                int scaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
                short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
                cachedScale = decodeFloat16Fast(scaleFloat16);
                lastBlockIndex = blockIndex;
            }
            scale = cachedScale;

            // Read quantized value
            byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);

            // Dequantize and accumulate
            result = fma(quantized * scale, that.get(j), result);
        }

        out.set(idx, result);
    }

    /**
     * Optimized float16 decoding using lookup table and bit manipulation
     */
    private static float    decodeFloat16Fast(short value) {
        // Split the components
        int sign = (value & 0x8000) >>> 15;
        int exp = (value & 0x7C00) >>> 10;
        int frac = value & 0x03FF;

        // Handle special cases with direct returns for common values
        if (exp == 0x1F) {
            return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        }

        if (exp == 0) {
            if (frac == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            // Optimize denormalized numbers with precomputed constant
            float result = frac * 5.9604645E-8f; // Precomputed 2^-24
            return sign == 0 ? result : -result;
        }

        // Normal case - optimize with fewer operations
        float result = 1.0f + (frac / 1024.0f);

        // Use bitshift instead of pow for integer powers of 2
        if (exp < 15) {
            int shift = 15 - exp;
            result /= (1 << shift);
        } else {
            int shift = exp - 15;
            result *= (1 << shift);
        }

        return sign == 0 ? result : -result;
    }

    /**
     * Fused multiply-add operation that maps to OpenCL's native fma
     * This will optimize to the fma instruction in OpenCL
     */
    private static float fma(float a, float b, float c) {
        return a * b + c;
    }

    private static float decodeFloat16(short value) {
        int sign = (value & 0x8000) >>> 15;
        int exp = (value & 0x7C00) >>> 10;
        int frac = value & 0x03FF;

        // Handle special cases
        if (exp == 0x1F) {
            return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        }
        if (exp == 0) {
            if (frac == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            float result = frac * pow2(-24);
            return sign == 0 ? result : -result;
        }

        float result = 1.0f + (frac / 1024.0f);
        result *= pow2(exp - 15);
        return sign == 0 ? result : -result;
    }

    public static void matmulTornadoQ8Optimized(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
        final int BLOCK_SIZE = 32; // Block size used in quantization
        final int BYTES_PER_BLOCK = 2 + BLOCK_SIZE; // 2 bytes for scale + block_size bytes for values
        final int UNROLL_FACTOR = 16; // Increased unroll factor for better performance
        final int VECTOR_SIZE = 4; // Process 4 elements at once with vectorization

        int idx = context.globalIdx;
        float result = 0f;
        int thisOffset = idx * dim1;

        // Cache last block index and scale to avoid redundant decoding
        int lastBlockIndex = -1;
        float cachedScale = 0f;

        // Early calculation of block boundaries to reduce in-loop calculations
        int numFullUnrolls = dim1 / UNROLL_FACTOR;
        int remainingStart = numFullUnrolls * UNROLL_FACTOR;

        // Pre-calculate first block index to potentially save work in the loop
        int firstIndex = thisOffset;
        int firstBlockIndex = firstIndex / BLOCK_SIZE;
        int firstBlockOffset = firstBlockIndex * BYTES_PER_BLOCK;

        // Initial scale calculation outside the loop
        int scaleByte1 = thisx.get(firstBlockOffset) & 0xFF;
        int scaleByte2 = thisx.get(firstBlockOffset + 1) & 0xFF;
        short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
        cachedScale = decodeFloat16Fast(scaleFloat16);
        lastBlockIndex = firstBlockIndex;

        // Main loop with increased unrolling
        for (int j = 0; j < numFullUnrolls; j++) {
            int baseIdx = j * UNROLL_FACTOR;

            // Process elements in groups of UNROLL_FACTOR
            for (int k = 0; k < UNROLL_FACTOR; k += VECTOR_SIZE) {
                // Process VECTOR_SIZE elements in each iteration
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    int index = thisOffset + baseIdx + k + v;
                    int blockIndex = index / BLOCK_SIZE;

                    // Only decode scale if we're in a new block
                    if (blockIndex != lastBlockIndex) {
                        int blockOffset = blockIndex * BYTES_PER_BLOCK;
                        int newScaleByte1 = thisx.get(blockOffset) & 0xFF;
                        int newScaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
                        short newScaleFloat16 = (short) ((newScaleByte2 << 8) | newScaleByte1);
                        cachedScale = decodeFloat16Fast(newScaleFloat16);
                        lastBlockIndex = blockIndex;
                    }

                    int withinBlockIndex = index % BLOCK_SIZE;
                    int blockOffset = blockIndex * BYTES_PER_BLOCK;

                    // Read quantized value
                    byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);

                    // Dequantize and accumulate
                    result = fma(quantized * cachedScale, that.get(baseIdx + k + v), result);
                }
            }
        }

        // Handle remaining elements
        for (int j = remainingStart; j < dim1; j++) {
            int index = thisOffset + j;
            int blockIndex = index / BLOCK_SIZE;

            // Only decode scale if we're in a new block
            if (blockIndex != lastBlockIndex) {
                int blockOffset = blockIndex * BYTES_PER_BLOCK;
                int scaleByte11 = thisx.get(blockOffset) & 0xFF;
                int scaleByte22 = thisx.get(blockOffset + 1) & 0xFF;
                short scaleFloat166 = (short) ((scaleByte22 << 8) | scaleByte11);
                cachedScale = decodeFloat16Fast(scaleFloat166);
                lastBlockIndex = blockIndex;
            }

            int withinBlockIndex = index % BLOCK_SIZE;
            int blockOffset = blockIndex * BYTES_PER_BLOCK;

            // Read quantized value
            byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);

            // Dequantize and accumulate
            result = fma(quantized * cachedScale, that.get(j), result);
        }

        out.set(idx, result);
    }

    public static void matmulTornadoQ8LocalMem(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
        final int BLOCK_SIZE = 32; // Block size used in quantization
        final int BYTES_PER_BLOCK = 2 + BLOCK_SIZE; // 2 bytes for scale + block_size bytes for values
        final int TS = 16; // Tile size for local memory

        // Thread identifiers
        int row = context.localIdx;
        int col = context.localIdy;
        int globalRow = TS * context.groupIdx + row;
        int globalCol = TS * context.groupIdy + col;

        // Allocate local memory for tiles
        float[] thatSub = context.allocateFloatLocalArray(TS * TS);
        float[] thisxSub = context.allocateFloatLocalArray(TS * TS);

        float result = 0f;

        // Loop over all tiles
        int numTiles = dim1 / TS;
        for (int t = 0; t < numTiles; t++) {
            // Load one tile of data into local memory with dequantization
            int tiledRow = TS * t + row;
            int tiledCol = TS * t + col;

            // Calculate global indices
            int thisIndex = (globalRow * dim1) + tiledCol;
            int thatIndex = (tiledRow * dim1) + globalCol;

            // Dequantize thisx data
            int thisBlockIndex = thisIndex / BLOCK_SIZE;
            int thisWithinBlockIndex = thisIndex % BLOCK_SIZE;
            int thisBlockOffset = thisBlockIndex * BYTES_PER_BLOCK;

            // Decode scale for thisx
            int scaleByte1 = thisx.get(thisBlockOffset) & 0xFF;
            int scaleByte2 = thisx.get(thisBlockOffset + 1) & 0xFF;
            short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
            float scale = decodeFloat16Fast(scaleFloat16);

            // Read quantized value and dequantize
            byte quantized = thisx.get(thisBlockOffset + 2 + thisWithinBlockIndex);
            thisxSub[col * TS + row] = quantized * scale;

            // Load that data directly (no quantization)
            thatSub[col * TS + row] = that.get(thatIndex);

            // Synchronize to make sure the tile is loaded
            context.localBarrier();

            // Perform matrix multiplication for a single tile
            for (int k = 0; k < TS; k++) {
                result = fma(thisxSub[k * TS + row], thatSub[col * TS + k], result);
            }

            // Synchronize before loading the next tile
            context.localBarrier();
        }

        // Handle edge case if dimension is not a multiple of tile size
        if (globalRow < dim1 && globalCol < dim1) {
            // Store the final result
            out.set(globalCol * dim1 + globalRow, result);
        }
    }

    /**
     * Compute 2^n efficiently
     */

    public static void matmul(ByteArray thisx, FloatArray that, FloatArray out, int dim0, int dim1) {
        final int BLOCK_SIZE = 32; // Assuming this is the block size used in quantization
        final int BYTES_PER_BLOCK = Float16.BYTES + BLOCK_SIZE; // 2 bytes for scale + block_size bytes for values

        IntStream.range(0, dim0).parallel().forEach(i -> {
            float result = 0f;
            int thisOffset = i * dim1;

            for (int j = 0; j < dim1; j++) {
                int index = thisOffset + j;
                // Calculate block position
                int blockIndex = index / BLOCK_SIZE;
                int withinBlockIndex = index % BLOCK_SIZE;
                int blockOffset = blockIndex * BYTES_PER_BLOCK;

                // Read scale (float16) for this block
                int scaleByte1 = thisx.get(blockOffset) & 0xFF;
                int scaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
                short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
                float scale = decodeFloat16(scaleFloat16);

                // Read quantized value
                byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);
                //                byte quantized = thisx.get(blockOffset + Float16.BYTES + withinBlockIndex);

                // Dequantize and multiply
                result += (quantized * scale) * that.get(j);
            }

            out.set(i, result);
        });
    }



    public static void copyToCache(FloatArray destKeyCache, FloatArray srcKey, FloatArray destValueCache, FloatArray srcValue, IntArray positioNlayer) {
        int destOffset = positioNlayer.get(2);
        for (@Parallel int i = 0; i < srcValue.getSize(); i++) {
            destKeyCache.set(destOffset + i, srcKey.get(i));
            destValueCache.set(destOffset + i, srcValue.get(i));
        }
    }

    public static void processHeadsParallel(
            FloatArray q, FloatArray key_cache, FloatArray value_cache, FloatArray xb,
            int nHeads, int headSize, int kvDim, int kvMul, int seqLen,
            IntArray positionNlayer, FloatArray wrapAtt) {

        int pos = positionNlayer.get(0);
        long loff = positionNlayer.get(3);

        // Parallelize computation across attention heads
        for (@Parallel int h = 0; h < nHeads; h++) {
            // Process each head in parallel
            processHeadTornado(q, key_cache, value_cache, xb, h, headSize, kvDim, kvMul, loff, pos, wrapAtt);
        }
    }

    private static void processHeadTornado(
            FloatArray allQ, FloatArray key_cache, FloatArray value_cache, FloatArray allXb,
            int h, int headSize, int kvDim, int kvMul, long loff, int pos, FloatArray wrapAtt) {

        // Base index for this head's attention weights
        int headOffset = h * (pos + 1);

        // STEP 1: Calculate attention scores for all timesteps
        for (int t = 0; t <= pos; t++) {
            int kvHeadIdx = h / kvMul;
            int keyOffset = (int) (loff + t * kvDim + kvHeadIdx * headSize);

            float score = 0.0f;
            for (int i = 0; i < headSize; i++) {
                score += allQ.get(h * headSize + i) * key_cache.get(keyOffset + i);
            }
            score = score / TornadoMath.sqrt(headSize);

            // Store in attention buffer
            wrapAtt.set(headOffset + t, score);
        }

        // STEP 2: Find max score for softmax stability
        float maxScore = wrapAtt.get(headOffset);
        for (int t = 1; t <= pos; t++) {
            float val = wrapAtt.get(headOffset + t);
            if (val > maxScore) {
                maxScore = val;
            }
        }

        // STEP 3: Compute exponentials and sum
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) {
            int idx = headOffset + t;
            float expScore =  TornadoMath.exp(wrapAtt.get(idx) - maxScore);
            wrapAtt.set(idx, expScore);
            sum += expScore;
        }

        // STEP 4: Normalize
        float normFactor = (sum > 0.0f) ? (1.0f / sum) : (1.0f / (pos + 1));
        for (int t = 0; t <= pos; t++) {
            int idx = headOffset + t;
            wrapAtt.set(idx, wrapAtt.get(idx) * normFactor);
        }

        // STEP 5: Compute weighted sum of values for each dimension
        for (int i = 0; i < headSize; i++) {
            float weightedSum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                int kvHeadIdx = h / kvMul;
                int valueOffset = (int) (loff + t * kvDim + kvHeadIdx * headSize);
                weightedSum += wrapAtt.get(headOffset + t) * value_cache.get(valueOffset + i);
            }
            allXb.set(h * headSize + i, weightedSum);
        }
    }

    public static void processHeadsParallel(
            FloatArray q, FloatArray key_cache, FloatArray value_cache, FloatArray xb,
            int nHeads, int headSize, int kvDim, int kvMul, int seqLen,
            IntArray positionNlayer) {

        int pos = positionNlayer.get(0);
        long loff = positionNlayer.get(3);

        // Parallelize computation across attention heads
        for (@Parallel int h = 0; h < nHeads; h++) {
            // Process each head in parallel using the memory-efficient approach
            processHeadTornadoOptimized(q, key_cache, value_cache, xb, h, headSize, kvDim, kvMul, loff, pos);
        }
    }

    private static void processHeadTornadoOptimized(
            FloatArray allQ, FloatArray key_cache, FloatArray value_cache, FloatArray allXb,
            int h, int headSize, int kvDim, int kvMul, long loff, int pos) {

        // Store only the attention scores, which is much smaller than the full wrapAtt array
        float[] scores = new float[pos + 1];
        float maxScore = -Float.MAX_VALUE;
        float[] accum = new float[headSize]; // Store weighted sum result

        // Initialize accumulators
        for (int i = 0; i < headSize; i++) {
            accum[i] = 0.0f;
        }

        // STEP 1: Compute dot products and track max score
        for (int t = 0; t <= pos; t++) {
            int kvHeadIdx = h / kvMul;
            int keyOffset = (int) (loff + t * kvDim + kvHeadIdx * headSize);
            float score = 0.0f;

            // Calculate dot product
            for (int i = 0; i < headSize; i++) {
                score += allQ.get(h * headSize + i) * key_cache.get(keyOffset + i);
            }
            score = score / TornadoMath.sqrt(headSize);
            scores[t] = score;

            // Update running max
            if (score > maxScore) {
                maxScore = score;
            }
        }

        // STEP 2+3+4: Apply softmax and accumulate weighted sum in one pass
        float sumExp = 0.0f;
        for (int t = 0; t <= pos; t++) {
            int kvHeadIdx = h / kvMul;
            int valueOffset = (int) (loff + t * kvDim + kvHeadIdx * headSize);

            // Apply exp with max trick
            float expScore = TornadoMath.exp(scores[t] - maxScore);
            sumExp += expScore;

            // Accumulate weighted values
            for (int i = 0; i < headSize; i++) {
                accum[i] += expScore * value_cache.get(valueOffset + i);
            }
        }

        // STEP 5: Normalize the weighted sum
        float invSumExp = (sumExp > 0.0f) ? (1.0f / sumExp) : (1.0f / (pos + 1));
        for (int i = 0; i < headSize; i++) {
            allXb.set(h * headSize + i, accum[i] * invSumExp);
        }
    }


    public static void reductionOneBlockForLogits(KernelContext context, FloatArray output, FloatArray x,
            int size, float ermsNorm, int localMemSize) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupId = context.groupIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory for reduction
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        // Load input value and compute square
        if (gid < size) {
            localX[lid] = x.get(gid);
            localX[lid] = localX[lid] * localX[lid];
        } else {
            localX[lid] = 0.0f;
        }

        // Perform parallel reduction within the work group
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        // Each workgroup stores its partial sum in a different location
        if (lid == 0) {
            // Store the partial sum from each workgroup
            output.set(size + groupId + 1, localX[0]);
        }

        // Only the first thread in the first workgroup computes the final normalization factor
        if (gid == 0) {
            // Combine partial sums from all workgroups
            float ss = 0.0f;
            for (int i = 1; i <= 8; i++) {  // Assuming 8 workgroups
                ss += output.get(i);
            }

            ss /= size;
            ss += ermsNorm;
            ss = 1.0f / TornadoMath.sqrt(ss);
            output.set(0, ss);  // Store the final scale factor
        }
    }

    // Second task: Apply normalization with weights
    public static void applyNormForLogits(KernelContext context, FloatArray output, FloatArray weights,
            int size, FloatArray tempLogits) {
//        int gid = context.globalIdx;
//
//        if (gid < size) {
//            // Apply normalization with weights
//            float ss = output.get(size);  // Get scale factor stored at position size
//            float val = output.get(gid);
//            output.set(gid, weights.get(gid) * (ss * val));
//        }
        int gid = context.globalIdx;

        if (gid < size) {
            // Get scale factor from position 'size' (not hardcoded offset)
            float scaleFactor = tempLogits.get(0);

            // Load the input value from the correct position
            float val = output.get(gid);

            // Load the weight from the correct position
            float w = weights.get(gid);

            // Apply normalization: weight * (scale * value)
            float normalized = w * (scaleFactor * val);

            // Store the result back to the same position
            output.set(gid, normalized);
        }
    }

    public static void projectionTwoOptimized(KernelContext context,  FloatArray x, FloatArray hb, FloatArray w,
            int n, int d, IntArray positionAndLayer, int localWorkGroupSize) {
        int rowId = context.groupIdx;      // One row per workgroup
        int localId = context.localIdx;    // Thread ID within workgroup
        int localSize = localWorkGroupSize;

        if (rowId >= d) {
            return;
        }

        float[] localSum = context.allocateFloatLocalArray(localSize);
        int layer = positionAndLayer.get(1);
        int layerOffset = layer * n * d;
        int rowOffset = layerOffset + rowId * n;
        float partialSum = 0.0f;

        for (int j = localId; j < n; j += localSize) {
            int matrixIdx = rowOffset + j;
            int vecIdx = j;
            float matrixVal = w.get(matrixIdx);
            float vecVal = x.get(vecIdx);
            partialSum += matrixVal * vecVal;
        }

        localSum[localId] = partialSum;
        context.localBarrier();

        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        if (localId == 0) {
            float sum = localSum[0];
            int outIdx = rowId;
            float hbVal = hb.get(outIdx);
            float silu = hbVal * (1.0f / (1.0f + TornadoMath.exp(-hbVal)));
            float result = silu * sum;
            hb.set(outIdx, result);
        }
    }
}
