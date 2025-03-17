package com.example.tornadovm;

import com.example.core.model.GGMLType;
import com.example.core.types.Float16;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.IntStream;

public class TornadoVMCompute {
    public static final boolean TORNADOVM = Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"));
    public static final long WORKGROUP = Long.parseLong(System.getProperty("llama.workgroup", "16"));

    public TornadoVMCompute() {
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

    public static void ropeRotation(KernelContext context, IntArray positionNlayer,
            FloatArray sq, FloatArray sk, int kv_dim, int head_size) {
        int i = context.globalIdx * 2;
        if (i >= kv_dim) {
            return;
        }

        int head_dim = i % head_size;
        float freq = 1.0f / TornadoMath.pow(10000.0f, head_dim / (float) head_size);
        float val = positionNlayer.get(0) * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        float v0 = sq.get(0);
        float v1 = sq.get(1);
        sq.set(0, v0 * fcr - v1 * fci);
        sq.set(0 + 1, v0 * fci + v1 * fcr);
        for (int v = 1; v < rotn; v++) {
            float v0z = sk.get(i);
            float v1z = sk.get(i + 1);
            sk.set(i, v0z * fcr - v1z * fci);
            sk.set(i + 1, v0z * fci + v1z * fcr);
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

    public static void matmudl(ByteArray thisx, FloatArray that, FloatArray out, int dim0, int dim1) {
        IntStream.range(0, dim0).parallel().forEach(i -> {
            float result = 0f;
            int thisOffset = i * dim1;
            for (int j = 0; j < dim1; j++) {
                //                result += thisx.get(thisOffset + j) * that.get(j);
                result += getFloatFromByteArray(thisOffset + j, thisx) * that.get(j);
            }
            out.set(i, result);
        });
    }

    //    public static void reduceSquareSums(KernelContext context, FloatArray x, FloatArray reduce) {
    //        int globalIdx = context.globalIdx;
    //        int localIdx = context.localIdx;
    //        int localGroupSize = context.localGroupSizeX;
    //        int groupID = context.groupIdx; // Expose Group ID
    //
    //        float[] localA = context.allocateFloatLocalArray((int) 256);
    //        localA[localIdx] = x.get(globalIdx) * x.get(globalIdx);
    //        for (int stride = (localGroupSize / 2); stride > 0; stride /= 2) {
    //            context.localBarrier();
    //            if (localIdx < stride) {
    //                localA[localIdx] += localA[localIdx + stride];
    //            }
    //        }
    //        if (localIdx == 0) {
    //            reduce.set(groupID, localA[0]);
    //        }
    //    }

    //    public static void finalSum(KernelContext context, FloatArray reduce, int size, float eps) {
    //        int globalIdx = context.globalIdx;
    //
    //        float sum = 0.0f;
    //        if (globalIdx == 0) {
    //            for (int i = 0; i < size; i++) {
    //                sum += reduce.get(i);
    //            }
    //        }
    //
    //        float ss = sum / (float) size;
    //        ss += eps;
    //        ss = 1.0f / TornadoMath.sqrt(ss);
    //        reduce.set(0, ss);
    //    }

    public static void normalizeAndScale(KernelContext context, FloatArray x, FloatArray weight, FloatArray scalingFactorBuffer, int size) {

        int globalIdx = context.globalIdx;

        if (globalIdx < size) {
            float scaledValue = weight.get(globalIdx) * (scalingFactorBuffer.get(0) * x.get(globalIdx));
            x.set(globalIdx, scaledValue);
        }
    }

    public static void normalizeAndScale2(KernelContext context, FloatArray out, FloatArray x, FloatArray weight, FloatArray scalingFactorBuffer, int size) {

        int globalIdx = context.globalIdx;

        if (globalIdx < size) {
            float scaledValue = weight.get(globalIdx) * (scalingFactorBuffer.get(0) * x.get(globalIdx));
            out.set(globalIdx, scaledValue);
        }
    }

//    public static void addInPlace(FloatArray input, FloatArray output) {
//        for (@Parallel int i = 0; i < input.getSize(); i++) {
//            // Perform element-wise addition
//            float result = output.get(i) + input.get(i);
//            output.set(i, result);
//        }
//    }

//    public static void multiplyInPlace(FloatArray input, FloatArray output) {
//        for (@Parallel int i = 0; i < input.getSize(); i++) {
//            // Perform element-wise multiplication
//            float result = output.get(i) * input.get(i);
//            output.set(i, result);
//        }
//    }

//    public static void mapInPlace(FloatArray input) {
//        for (@Parallel int i = 0; i < input.getSize(); i++) {
//            // Apply the transformation: value -> value / (1.0 + exp(-value))
//            float value = input.get(i);
//            float result = value / (1.0f + TornadoMath.exp(-value));
//            input.set(i, result);
//        }
//    }

    public static void matrixVectorSimple(FloatArray x, FloatArray xout, FloatArray w, int n, int d) {
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            float val = 0f;
            for (int j = 0; j < xout.getSize(); j++) {
                val += w.get(i * n + j) * x.get(j);
            }
            xout.set(i, val);
        }
    }

    public static void matrixVectorSimple(FloatArray x, FloatArray xout, FloatArray w, int n, int d, int layerShift) {
        int wOffset = layerShift * d;  // Correctly locate the starting index for the desired layer

        for (@Parallel int i = 0; i < xout.getSize(); i++) {
            float val = 0f;
            for (int j = 0; j < x.getSize(); j++) {
                val += w.get(wOffset + i * n + j) * x.get(j); // Corrected access pattern
            }
            xout.set(i, val);
        }
    }

    public static void matrixVectorSimpleF15(FloatArray x, FloatArray xout, HalfFloatArray w, int n, int d) {
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            float val = 0f;
            for (int j = 0; j < xout.getSize(); j++) {
                val += w.get(i * n + j).getFloat32() * x.get(j);
            }
            xout.set(i, val);
        }
    }

    // < --------------------------------------------------- >

    public static void reduceSquareSums(KernelContext context, FloatArray a, FloatArray reduce, int localWorkGroupSize) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;
        int localGroupSize = context.localGroupSizeX;
        int groupID = context.groupIdx; // Expose Group ID

        float[] localA = context.allocateFloatLocalArray(localWorkGroupSize);
        localA[localIdx] = a.get(globalIdx) * a.get(globalIdx);
        for (int stride = (localGroupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (localIdx < stride) {
                localA[localIdx] += localA[localIdx + stride];
            }
        }
        if (localIdx == 0) {
            reduce.set(groupID, localA[0]);
        }
    }

//    public static void finalSum(KernelContext context, FloatArray reduce, int size, float eps) {
//        int globalIdx = context.globalIdx;
//
//        float sum = 0.0f;
//        if (globalIdx == 0) {
//            for (int i = 0; i < size; i++) {
//                sum += reduce.get(i);
//            }
//        }
//
//        float ss = sum / (float) size;
//        ss += eps;
//        ss = 1.0f / TornadoMath.sqrt(ss);
//        reduce.set(0, ss);
//    }

    public static void finalSum(KernelContext context, FloatArray reduce, int size, float eps) {
        int globalIdx = context.globalIdx;

        float sum = 0.0f;
        if (globalIdx == 0) {
            // Use reduce.getSize() instead of size
            for (int i = 0; i < reduce.getSize(); i++) {
                sum += reduce.get(i);
            }
        }

        float ss = sum / (float) size;  // Keep dividing by the original size
        ss += eps;
        ss = 1.0f / TornadoMath.sqrt(ss);
        reduce.set(0, ss);
    }

    public static void normalizeAndScale(KernelContext context, FloatArray out, FloatArray
            input, FloatArray weight, FloatArray scalingFactorBuffer, int size, IntArray positionNlayer) {

        int globalIdx = context.globalIdx;

        int layerOffset = positionNlayer.get(1) * size;

        if (globalIdx < size) {
            float scaledValue = weight.get(layerOffset + globalIdx) * (scalingFactorBuffer.get(0) * input.get(globalIdx));
            out.set(globalIdx, scaledValue);
        }
    }

    public static void matrixVectorSimple(KernelContext context, FloatArray x, FloatArray output, FloatArray weights, int n, int d) {
        int idx = context.globalIdx;

        if (idx < output.getSize()) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += weights.get(idx * n + j) * x.get(j);
            }
            output.set(idx, sum);
        }
    }


    public static void matrixVectorSimple(KernelContext context,
            FloatArray x, FloatArray output, FloatArray weights, int n, int d, IntArray posAndLayer) {
        int idx = context.globalIdx;

        if (idx < output.getSize()) {  // Ensure we don't go out of bounds
            int layer = posAndLayer.get(1);
            // Base offset for the current layer: layer * d * n
            // Each layer has a full d×n matrix
            int layerOffset = layer * d * n;

            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                // For each output idx, we need to do a dot product of the row idx with vector x
                // The weights are stored in row-major format
                sum += weights.get(layerOffset + idx * n + j) * x.get(j);
            }
            output.set(idx, sum);
        }
    }

    /**
     * Calculate attention scores between query and key vectors
     */
    public static void calculateAttentionScores(KernelContext context, IntArray positionNlayer, int seqLen, FloatArray query, FloatArray keyCache, FloatArray attScores, int kvDim, int kvMul, int headSize, int loff) {
        int h = context.groupIdx;         // Head index
        int threadId = context.localIdx;  // Thread ID within work group
        int blockDim = context.localGroupSizeX;  // Work group size

        // Get the query vector offset for this head
        int queryOffset = h * headSize;

        // Attention scores offset for this head
        int attOffset = h * seqLen;

        // Iterate over all timesteps, including the current one
        for (int t = threadId; t <= positionNlayer.get(1); t += blockDim) {
            // Get the key vector for this head and at this timestep
            int keyOffset = loff + t * kvDim + (h / kvMul) * headSize;

            // Calculate the attention score as the dot product of query and key
            float score = 0.0f;
            for (int i = 0; i < headSize; i++) {
                score += query.get(queryOffset + i) * keyCache.get(keyOffset + i);
            }

            // Scale by sqrt(head_size)
            score /= Math.sqrt(headSize);

            // Save the score to the attention buffer
            attScores.set(attOffset + t, score);
        }
    }

    /**
     * Find maximum attention score for numerical stability in softmax
     */
    public static void findMaxAttentionScores(KernelContext context, IntArray positionNlayer, int seqLen, FloatArray attScores, FloatArray maxValues, int workGroupSize) {
        int h = context.groupIdx;         // Head index
        int threadId = context.localIdx;  // Thread ID within work group
        int blockDim = context.localGroupSizeX;  // Work group size

        // Attention scores offset for this head
        int attOffset = h * seqLen;

        // Find the maximum value for numerical stability
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int t = threadId; t <= positionNlayer.get(1); t += blockDim) {
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

    public static void calculateExpAndSum(KernelContext context, IntArray positionNlayer, int seqLen, FloatArray attScores, FloatArray maxValues, FloatArray expValues, FloatArray sumValues, int localWorkGroupSize) {
        int h = context.groupIdx;         // Head index
        int threadId = context.localIdx;  // Thread ID within work group
        int blockDim = context.localGroupSizeX;  // Work group size

        // Get max value for this head
        float maxVal = maxValues.get(h);

        // Attention scores and exp values offset for this head
        int attOffset = h * seqLen;
        int expOffset = h * seqLen;

        // Compute exp(score - max) and thread-local sum
        float expSum = 0.0f;
        for (int t = threadId; t <= positionNlayer.get(1); t += blockDim) {
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

        // Normalize values and write back to attention scores
        for (int t = threadId; t <= positionNlayer.get(1); t += blockDim) {
            float normalizedValue = expValues.get(expOffset + t) / sum;
            attScores.set(attOffset + t, normalizedValue);
        }
    }

    public static void computeWeightedSum(KernelContext context, IntArray positionNlayer, int seqLen, FloatArray attScores, FloatArray valueCache, FloatArray output, int kvDim, int kvMul, int headSize, int loff) {
        int h = context.groupIdx;         // Head index
        int threadId = context.localIdx;  // Thread ID within work group
        int blockDim = context.localGroupSizeX;  // Work group size

        // Attention scores offset for this head
        int attOffset = h * seqLen;

        // Output offset for this head
        int outputOffset = h * headSize;

        // Calculate weighted sum for each head dimension
        for (int i = threadId; i < headSize; i += blockDim) {
            float val = 0.0f;
            for (int t = 0; t <= positionNlayer.get(1); t++) {
                // Get the value vector for this head and timestep
                int valueOffset = loff + t * kvDim + (h / kvMul) * headSize;

                // Get the attention weight for this timestep
                float a = attScores.get(attOffset + t);

                val += a * valueCache.get(valueOffset + i);
            }
            output.set(outputOffset + i, val);
        }

        // Make sure all threads finish writing their outputs
        context.localBarrier();
    }

    /**
     * Matrix-vector multiplication using KernelContext
     */
    public static void matrixVectorMultiply(KernelContext context, FloatArray x, FloatArray output, FloatArray weights, int n,
            int d, IntArray positionNlayer) {
        int idx = context.globalIdx;

        int layerOffset = positionNlayer.get(1) * n;

        if (idx < d) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                if (j < x.getSize() && (idx * n + j) < weights.getSize()) {
                    sum += weights.get(idx * layerOffset + j) * x.get(j);
                }
            }

            output.set(idx, sum);
        }
    }

    /**
     * In-place addition using KernelContext
     */
    public static void addInPlace(KernelContext context, FloatArray input, FloatArray output) {
        int idx = context.globalIdx;

        if (idx < Math.min(input.getSize(), output.getSize())) {
            output.set(idx, output.get(idx) + input.get(idx));
        }
    }

    /**
     * SiLU activation function using KernelContext
     */
    public static void siluActivation(KernelContext context, FloatArray input) {
        int idx = context.globalIdx;

        if (idx < input.getSize()) {
            float value = input.get(idx);
            float result = value / (1.0f + TornadoMath.exp(-value));
            input.set(idx, result);
        }
    }

    /**
     * Element-wise multiplication using KernelContext
     */
    public static void elementMultiply(KernelContext context, FloatArray input, FloatArray output) {
        int idx = context.globalIdx;

        if (idx < Math.min(input.getSize(), output.getSize())) {
            output.set(idx, output.get(idx) * input.get(idx));
        }
    }

    public static void emptyTaskToForceCopyIn(FloatArray a) {
        // Empty method with a dummy condition to avoid optimization
        if (a.getSize() > 100000) {
        }
    }

}
