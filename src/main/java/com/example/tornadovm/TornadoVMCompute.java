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
    public static final boolean TORNADOVM = Boolean.parseBoolean(System.getProperty("use.tornadovm", "true"));

    public TornadoVMCompute() {
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

        int layerOffset = positionAndLayer.get(1) * d * n;  // l * dim * dim for example

        for (@Parallel int i = 0; i < d; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += w.get(layerOffset + i * n + j) * x.get(j);
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
            hb2.set(i, val);
        }
    }

    /**
     * Reductions launched in a single thread-block
     *
     * @param context
     * @param output
     * @param x
     */
    public static void reductionOneBlock(KernelContext context, FloatArray output, FloatArray x, int localRMS, float rmsNormEps) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupSize = context.localGroupSizeX;
        float[] localX = context.allocateFloatLocalArray(localRMS);
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

    public static void ropeRotation(KernelContext context, IntArray positionNlayer, FloatArray sq, FloatArray sk, int kv_dim, int head_size) {
        int i = context.globalIdx * 2;

        int head_dim = i % head_size;
        float freq = 1.0f / TornadoMath.pow(10000.0f, head_dim / (float) head_size);
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
        final int BLOCK_SIZE = 32; // Assuming this is the block size used in quantization
        final int BYTES_PER_BLOCK = Float16.BYTES + BLOCK_SIZE; // 2 bytes for scale + block_size bytes for values

        int idx = context.globalIdx;

        float result = 0f;
        int thisOffset = idx * dim1;

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

            // Dequantize and multiply
            result += (quantized * scale) * that.get(j);
        }

        out.set(idx, result);

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

    private static float getFloatFromByteArray(int index, ByteArray data) {
        // Direct read of two bytes at the index position - no multiplication by BYTES_PER_FLOAT16
        int byte1 = data.get(index) & 0xFF;
        int byte2 = data.get(index + 1) & 0xFF;
        short float16Value = (short) ((byte2 << 8) | byte1);

        return decodeFloat16(float16Value);
    }


//    private static void findMax(
//            KernelContext context,
//            IntArray positionNlayer,
//            FloatArray maxVal,
//            FloatArray att,
//            int localSize
//    ) {
//        int position = positionNlayer.get(0);
//        int gid = context.globalIdx;
//        int lid = context.localIdx;
//        int groupSize = context.localGroupSizeX;
//        float[] localX = context.allocateFloatLocalArray(localSize);
//        localX[lid] = att.get(gid);
//        // Step 2: Reduce to find global maximum across all threads
//        float[] localMaxes = context.allocateFloatLocalArray(blockDim);
//        localMaxes[tid] = maxVal;
//
//        // Parallel reduction to find global maximum
//        for (int stride = blockDim / 2; stride > 0; stride /= 2) {
//            context.localBarrier();
//            if (tid < stride) {
//                localMaxes[tid] = Math.max(localMaxes[tid], localMaxes[tid + stride]);
//            }
//        }
//
//
//        if (lid == 0) {
//            // store max
//            maxVal.set(0, localX[0]);
//        }
//
//        int tid = context.localIdx;         // Thread ID within work group
//        int blockDim = context.localGroupSizeX;  // Work group size
//        int position = positionNlayer.get(0);
//        int size = position + 1;  // Equivalent to size in CUDA version
//
//        int attOffset = headIdx * seqLen;  // Offset for this head
//
//        // Step 1: Each thread finds max in its assigned section
//        float maxVal = Float.NEGATIVE_INFINITY;
//
//        // Initialize with thread's first assigned value
//        if (tid < size) {
//            maxVal = attScores.get(attOffset + tid);
//        }
//
//        // Scan through remaining assigned values
//        for (int i = tid + blockDim; i < size; i += blockDim) {
//            float current = attScores.get(attOffset + i);
//            maxVal = Math.max(maxVal, current);
//        }
//
//
//    }



    private static void expSum(KernelContext context, FloatArray output, FloatArray x) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupSize = context.localGroupSizeX;
        float[] localX = context.allocateFloatLocalArray(1024);
        localX[lid] = x.get(gid);
        float max = output.get(0);
        localX[lid] = TornadoMath.exp(localX[lid] - max);
        x.set(gid, localX[lid]);
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        if (lid == 0) {
            // final sum stored in ID 0
            output.set(0, localX[0]);
        }
    }

    private static void norm(KernelContext context, FloatArray temp, FloatArray x) {
        int gid = context.globalIdx;
        float sum = temp.get(0);
        x.set(gid, x.get(gid) / sum);
    }


    /**
     * Calculate attention scores between query and key vectors
     */
    public static void calculateAttentionScores(KernelContext context, FloatArray query, FloatArray key, FloatArray attentionScores, IntArray positionAndLayer, int headSize, int numHeads,
            int contextLength) {
        int pos = positionAndLayer.get(0);
        int layer = positionAndLayer.get(1);
        int kvOffset = positionAndLayer.get(2);

        // Calculate head and position indices
        int head = context.globalIdx / contextLength;
        int targetPos = context.globalIdx % contextLength;

        // Bounds checking
        if (head >= numHeads || targetPos >= contextLength) {
            return;
        }

        // Calculate offsets for query and key
        int queryOffset = (layer * numHeads + head) * headSize;
        int keyOffset = kvOffset + (head * headSize);

        // Compute dot product with bounds checking
        float score = 0.0f;
        for (int i = 0; i < headSize; i++) {
            if (queryOffset + i < query.getSize() && keyOffset + i < key.getSize()) {
                score += query.get(queryOffset + i) * key.get(keyOffset + i);
            }
        }

        // Scale the attention score
        score = score / (float) Math.sqrt(headSize);

        // Store the result with bounds checking
        int scoreIndex = head * contextLength + targetPos;
        if (scoreIndex < attentionScores.getSize()) {
            attentionScores.set(scoreIndex, score);
        }
    }

    /**
     * Find maximum attention score for numerical stability in softmax
     */
    public static void findMaxAttentionScoress(KernelContext context,
            IntArray positionNlayer,
            int seqLen, FloatArray attScores,
            FloatArray maxValues, int workGroupSize) {
        int h = context.groupIdx;         // Head index
        int threadId = context.localIdx;  // Thread ID within work group
        int blockDim = context.localGroupSizeX;  // Work group size

        // Attention scores offset for this head
        int attOffset = h * seqLen;

        // Find the maximum value for numerical stability
        float maxVal = Float.NEGATIVE_INFINITY;
        int position = positionNlayer.get(0) + 1;

        for (int t = threadId; t < position; t += blockDim) {
            maxVal = Math.max(maxVal, attScores.get(attOffset + t));
        }

        // Parallel reduction to find global maximum
        float[] maxReduction = context.allocateFloatLocalArray(workGroupSize); //TODO: ISSUES
        maxReduction[threadId] = maxVal;

        for (int stride = blockDim / 2; stride > 0; stride /= 2) {
            context.localBarrier();
            if (threadId < stride) {
                maxReduction[threadId] = Math.max(maxReduction[threadId], maxReduction[threadId + stride]);
            }
        }

        // Thread 0 in each work group writes the max value
        if (threadId == 0) {
            maxValues.set(h, maxReduction[0]);
        }
    }

    public static void calculateExpAndSum(KernelContext context, IntArray positionNlayer, int seqLen, FloatArray attScores, FloatArray maxValues, FloatArray expValues, FloatArray sumValues,
            int localWorkGroupSize) {
        int h = context.groupIdx;         // Head index
        int threadId = context.localIdx;  // Thread ID within work group
        int blockDim = context.localGroupSizeX;  // Work group size

        // Get max value for this head
        float maxVal = maxValues.get(h);

        // Attention scores and exp values offset for this head
        int attOffset = h * seqLen;
        int expOffset = h * seqLen;
        int position = positionNlayer.get(0) + 1;

        // Compute exp(score - max) and thread-local sum
        float expSum = 0.0f;
        for (int t = threadId; t < position; t += blockDim) {
            float score = attScores.get(attOffset + t);
            float expValue = (float) Math.exp(score - maxVal);
            expValues.set(expOffset + t, expValue);
            expSum += expValue;
        }

        // Ensure all exp values are computed before summing
        context.localBarrier();

        // Parallel reduction to get the total sum
        float[] sumReduction = context.allocateFloatLocalArray(localWorkGroupSize);
        sumReduction[threadId] = expSum;

        for (int stride = blockDim / 2; stride > 0; stride /= 2) {
            context.localBarrier();
            if (threadId < stride) {
                sumReduction[threadId] += sumReduction[threadId + stride];
            }
        }

        // Thread 0 in each work group writes the sum
        if (threadId == 0) {
            sumValues.set(h, sumReduction[0]);
        }

        // Ensure sum value is written before proceeding
        context.localBarrier();
    }

    /**
     * Normalize exponential values to get softmax probabilities
     */
    public static void normalizeSoftmax(KernelContext context, IntArray positionNlayer, int seqLen, FloatArray expValues, FloatArray sumValues, FloatArray attScores) {
        int h = context.groupIdx;         // Head index
        int threadId = context.localIdx;  // Thread ID within work group
        int blockDim = context.localGroupSizeX;  // Work group size

        // Get sum value for this head
        float sum = sumValues.get(h);

        // Exp values and attention scores offset for this head
        int expOffset = h * seqLen;
        int attOffset = h * seqLen;
        int position = positionNlayer.get(0) + 1;

        // Normalize values and write back to attention scores
        for (int t = threadId; t < position; t += blockDim) {
            float normalizedValue = expValues.get(expOffset + t) / sum;
            attScores.set(attOffset + t, normalizedValue);
        }
    }

    public static void computeWeightedSum(KernelContext context, IntArray positionNlayer, int seqLen, FloatArray attScores, FloatArray valueCache, FloatArray output, int kvDim, int kvMul,
            int headSize) {
        int h = context.groupIdx;         // Head index
        int threadId = context.localIdx;  // Thread ID within work group
        int blockDim = context.localGroupSizeX;  // Work group size

        // Attention scores offset for this head
        int attOffset = h * seqLen;

        // Output offset for this head
        int outputOffset = h * headSize;
        int position = positionNlayer.get(0) + 1;

        // Calculate weighted sum for each head dimension
        for (int i = threadId; i < headSize; i += blockDim) {
            float val = 0.0f;
            for (int t = 0; t < position; t++) {
                // Get the value vector for this head and timestep
                int valueOffset = positionNlayer.get(1) + t * kvDim + (h / kvMul) * headSize;

                // Get the attention weight for this timestep
                float a = attScores.get(attOffset + t);

                val += a * valueCache.get(valueOffset + i);
            }
            output.set(outputOffset + i, val);
        }

        // Make sure all threads finish writing their outputs
        context.localBarrier();
    }

    public static void copyToCache(FloatArray dest, FloatArray src, IntArray positioNlayer) {
        int destOffset = positioNlayer.get(2);
        for (@Parallel int i = 0; i < src.getSize(); i++) {
            dest.set(destOffset + i, src.get(i));
        }

    }
}
