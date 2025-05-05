package com.example.tornadovm;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

public class TransformerComputeKernels {

    public TransformerComputeKernels() {

    }

    public static void emptyTaskToForceCopyIn(FloatArray buffer) {
        float dummy = buffer.get(0);
        if (dummy > Float.MAX_VALUE) {
            buffer.set(0, dummy);
        }
    }

    public static void reductionOneBlockWithLayer(KernelContext context, FloatArray output, FloatArray x, int size, float ermsNorm, int localMemSize) {
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
            for (int i = 1; i <= (size / localMemSize); i++) {  // Assuming 8 workgroups
                ss += output.get(i);
            }

            ss /= size;
            ss += ermsNorm;
            ss = 1.0f / TornadoMath.sqrt(ss);
            output.set(0, ss);  // Store the final scale factor
        }
    }

    public static void reductionOneBlock2WithLayer(KernelContext context, FloatArray output, FloatArray x, FloatArray weights, FloatArray temp, IntArray positionAndLayer, int size) {
        int gid = context.globalIdx;

        if (gid < size) {
            // Get the layer offset from positionAndLayer
            int layerOffset = positionAndLayer.get(1) * size;

            // Apply normalization with the correct weight for this layer
            float ss = temp.get(0);
            output.set(gid, weights.get(layerOffset + gid) * (ss * x.get(gid)));
        }
    }

    public static void reductionOneBlock2WithLogits(KernelContext context, FloatArray output, FloatArray weights, FloatArray temp, IntArray positionAndLayer, int size) {
        int gid = context.globalIdx;

        if (gid < size) {
            // Apply normalization with the correct weight for this layer
            float ss = temp.get(0);
            output.set(gid, weights.get(gid) * (ss * output.get(gid)));
        }
    }

    public static void copyToCache(FloatArray destKeyCache, FloatArray srcKey, FloatArray destValueCache, FloatArray srcValue, IntArray positioNlayer) {
        int destOffset = positioNlayer.get(2);
        for (@Parallel int i = 0; i < srcValue.getSize(); i++) {
            destKeyCache.set(destOffset + i, srcKey.get(i));
            destValueCache.set(destOffset + i, srcValue.get(i));
        }
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

    public static void processHeadsParallel(FloatArray q, FloatArray key_cache, FloatArray value_cache, FloatArray xb, int nHeads, int headSize, int kvDim, int kvMul, int seqLen,
            IntArray positionNlayer, FloatArray wrapAtt) {

        int pos = positionNlayer.get(0);
        long loff = positionNlayer.get(3);

        // Parallelize computation across attention heads
        for (@Parallel int h = 0; h < nHeads; h++) {
            // Process each head in parallel
            processHeadTornado(q, key_cache, value_cache, xb, h, headSize, kvDim, kvMul, loff, pos, wrapAtt);
        }
    }

    private static void processHeadTornado(FloatArray allQ, FloatArray key_cache, FloatArray value_cache, FloatArray allXb, int h, int headSize, int kvDim, int kvMul, long loff, int pos,
            FloatArray wrapAtt) {

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
            float expScore = TornadoMath.exp(wrapAtt.get(idx) - maxScore);
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

    public static void matrixVectorGeneric(KernelContext context, FloatArray x, FloatArray hb, FloatArray w, int n, int d, IntArray positionAndLayer, int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= d) {
            return;
        }

        float sum = matrixVectorRowMajorOptimized(context, localSize, x, w, n, d, positionAndLayer);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            hb.set(rowId, sum);
        }
    }

    public static void matrixVectorGenericWithResidual(KernelContext context, FloatArray x, FloatArray hb, FloatArray w, int n, int d, IntArray positionAndLayer, int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= d) {
            return;
        }

        float sum = matrixVectorRowMajorOptimized(context, localSize, x, w, n, d, positionAndLayer);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float result = hb.get(rowId) + sum;
            hb.set(rowId, result);
        }
    }

    public static void fusedFeedForwardWithSiLUAndGLUActivation(KernelContext context, FloatArray x, FloatArray hb, FloatArray w1, FloatArray w3, int n, int d, IntArray positionAndLayer,
            int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= d) {
            return;
        }

        float sum1 = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, w1, n, d, positionAndLayer);
        float sum3 = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, w3, n, d, positionAndLayer);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float silu = siluActivation(sum1);  // Using the new SiLU method
            float result = silu * sum3;
            hb.set(rowId, result);
        }
    }

    public static float siluActivation(float x) {
        return x * (1.0f / (1.0f + TornadoMath.exp(-x)));
    }

    public static float matrixVectorRowMajorOptimized(KernelContext context, int localSize, FloatArray x, FloatArray w, int n, int d, IntArray positionAndLayer) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localSize);

        // Calculate offsets based on layer
        int layer = positionAndLayer.get(1);
        int layerOffset = layer * n * d;
        int rowOffset = layerOffset + rowId * n;

        // Each thread calculates partial dot product
        float partialSum = 0.0f;
        for (int j = localId; j < n; j += localSize) {
            int matrixIdx = rowOffset + j;
            partialSum += w.get(matrixIdx) * x.get(j);
        }

        // Store partial sum in local memory
        localSum[localId] = partialSum;
        context.localBarrier();

        // Parallel reduction within workgroup
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        return localSum[0];
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

    private static float fma(float a, float b, float c) {
        return a * b + c;
    }

}
