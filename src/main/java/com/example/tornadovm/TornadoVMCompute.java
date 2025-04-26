package com.example.tornadovm;

import com.example.core.model.GGMLType;

import com.example.core.types.Float16;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

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
    public static void siluElemWiseMulActivation(int hidenDimSize, FloatArray hb, FloatArray hb2) {
        for (@Parallel int i = 0; i < hidenDimSize; i++) {
            float val = hb.get(i);
            val *= (1.0f / (1.0f + TornadoMath.exp(-val)));
            val *= hb2.get(i);
            hb.set(i, val);
        }
    }
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
    private static float decodeFloat16Fast(short value) {
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

}
