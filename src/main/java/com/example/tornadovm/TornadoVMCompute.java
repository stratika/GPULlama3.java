package com.example.tornadovm;

import com.example.core.model.GGMLType;
import com.example.core.model.tensor.FloatTensor;
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
     * First step: Compute square sums in parallel and store results per work group
     */
    public static void reduceSquareSums(KernelContext context, FloatArray partialSums, FloatArray input, int localSize) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;
        int localGroupSize = context.localGroupSizeX;
        int groupID = context.groupIdx;

        // Allocate local memory for reduction
        float[] localData = context.allocateFloatLocalArray(localSize);

        // Compute squares and store in local memory
        if (globalIdx < input.getSize()) {
            float val = input.get(globalIdx);
            localData[localIdx] = val * val;
        } else {
            localData[localIdx] = 0.0f;
        }

        // Perform parallel reduction within work group
        for (int stride = (localGroupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (localIdx < stride) {
                localData[localIdx] += localData[localIdx + stride];
            }
        }

        // Store partial sum for this work group
        if (localIdx == 0) {
            partialSums.set(groupID, localData[0]);
        }
    }


    public static void finalSum(FloatArray reduce, int size, float eps) {
        float sum = 0.0f;

        for (int i = 0; i < reduce.getSize(); i++) {
            sum += reduce.get(i);
        }

        float ss = sum / (float) size;
        ss += eps;
        ss = 1.0f / TornadoMath.sqrt(ss);
        reduce.set(0, ss);
    }



    /**
     * Third step: Apply normalization and scaling using the computed scale factor
     */
    public static void normalizeAndScale(KernelContext context, FloatArray output, FloatArray input,
            FloatArray weights, FloatArray scaleFactorBuffer,
            IntArray positionAndLayer, int size) {
        int globalIdx = context.globalIdx;

        if (globalIdx < size) {
            // Get the layer offset for weights
            int layerOffset = positionAndLayer.get(1) * size;

            // Get the scale factor computed in the second step
            float scaleFactor = scaleFactorBuffer.get(0);

            // Apply normalization and scaling
            float inputVal = input.get(globalIdx);
            float weightVal = weights.get(layerOffset + globalIdx);
            float normalizedVal = weightVal * (scaleFactor * inputVal);

            output.set(globalIdx, normalizedVal);
        }
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

    public static void forcePropagationOneArray(FloatArray x) {
        x.set(0, x.get(0));
    }

    public static void forcePropagationOneArray(IntArray x) {
        x.set(0, x.get(0));
    }

    public static void forcePropagationTwoArrays(FloatArray x, FloatArray y) {
        x.set(0, x.get(0));
        y.set(0, y.get(0));
    }

    public static void forcePropagationTwoArrays(FloatArray x, IntArray y) {
        x.set(0, x.get(0));
        y.set(0, y.get(0));
    }

    public static void forcePropagationThreeArrays(FloatArray x, FloatArray y, FloatArray z) {
        x.set(0, x.get(0));
        y.set(0, y.get(0));
        z.set(0, z.get(0));
    }

    public static void forcePropagationThreeArrays(FloatArray x, FloatArray y, IntArray z) {
        x.set(0, x.get(0));
        y.set(0, y.get(0));
        z.set(0, z.get(0));
    }

    public static void forcePropagationFourArrays(FloatArray x, FloatArray y, FloatArray z, FloatArray w) {
        x.set(0, x.get(0));
        y.set(0, y.get(0));
        z.set(0, z.get(0));
        w.set(0, w.get(0));
    }

    public static void forcePropagationFiveArrays(FloatArray x, FloatArray y, FloatArray z, FloatArray w, FloatArray cv) {
        x.set(0, x.get(0));
        y.set(0, y.get(0));
        z.set(0, z.get(0));
        w.set(0, w.get(0));
        cv.set(0, cv.get(0));
    }

    public static void forcePropagationSixArrays(FloatArray x, FloatArray y, FloatArray z, FloatArray w, FloatArray cv, FloatArray xyz) {
        x.set(0, x.get(0));
        y.set(0, y.get(0));
        z.set(0, z.get(0));
        w.set(0, w.get(0));
        cv.set(0, cv.get(0));
        xyz.set(0, xyz.get(0));
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
             FloatArray wrapAtt, int pos, int layer) {

//        int pos = positionNlayer.get(0);
//        int layer = positionNlayer.get(1);
        long loff = layer * seqLen * kvDim; // layer offset into KV cache

        // Parallelize computation across attention heads
        for (int h = 0; h < nHeads; h++) {
            // Process each head in parallel
            processHeadTornado(q, key_cache, value_cache, xb, h, headSize, kvDim, kvMul, loff, pos, wrapAtt);
        }
    }

    public static void processHeadsParallel(
            FloatArray q, FloatArray key_cache, FloatArray value_cache, FloatArray xb,
            int nHeads, int headSize, int kvDim, int kvMul, int seqLen,
            IntArray positionNlayer, FloatArray wrapAtt) {

                int pos = positionNlayer.get(0);
                int layer = positionNlayer.get(1);
        long loff = layer * seqLen * kvDim; // layer offset into KV cache

        // Parallelize computation across attention heads
        for (int h = 0; h < nHeads; h++) {
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

//    private static void processHeadTornado(
//            FloatArray allQ, FloatArray key_cache, FloatArray value_cache, FloatArray allXb,
//            int h, int headSize, int kvDim, int kvMul, long loff, int pos, FloatArray wrapAtt) {
//
//        // Base index for this head's attention weights
//        int headOffset = h * (pos + 1);
//
//        // Local arrays for intermediate calculations
//        FloatArray scores = new FloatArray(pos + 1);
//
//        // STEP 1: Calculate attention scores for all timesteps
//        for (int t = 0; t <= pos; t++) {
//            int kvHeadIdx = h / kvMul;
//            int keyOffset = (int)(loff + t * kvDim + kvHeadIdx * headSize);
//
//            float score = 0.0f;
//            for (int i = 0; i < headSize; i++) {
//                score += allQ.get(h * headSize + i) * key_cache.get(keyOffset + i);
//            }
//            score /= (float)Math.sqrt(headSize);
//
//            // Store in local array
//            scores.set(t ,score);
//
//            // Also store in attention buffer
//            wrapAtt.set(headOffset + t, score);
//        }
//
//        // STEP 2: Apply softmax to attention scores
//        // Find maximum score for numerical stability
//        float maxScore = scores.get(0);
//        for (int t = 1; t <= pos; t++) {
//            if (scores.get(t) > maxScore) {
//                maxScore =  scores.get(t);
//            }
//        }
//
//        // Compute exponentials and sum
//        float sum = 0.0f;
//        for (int t = 0; t <= pos; t++) {
//            scores.set (t, (float)Math.exp(scores.get(t) - maxScore));
//            sum += scores.get(t);
//        }
//
//        // Normalize and store back to global array
//        for (int t = 0; t <= pos; t++) {
//            float normalizedScore = (sum > 0.0f) ? (scores.get(t) / sum) : (1.0f / (pos + 1));
//            wrapAtt.set(headOffset + t, normalizedScore);
//        }
//
//        // STEP 3: Compute weighted sum of values
//        float[] headOutput = new float[headSize];
//        for (int i = 0; i < headSize; i++) {
//            headOutput[i] = 0.0f;
//        }
//
//        for (int t = 0; t <= pos; t++) {
//            int kvHeadIdx = h / kvMul;
//            int valueOffset = (int)(loff + t * kvDim + kvHeadIdx * headSize);
//            float weight = wrapAtt.get(headOffset + t);
//
//            for (int i = 0; i < headSize; i++) {
//                headOutput[i] += weight * value_cache.get(valueOffset + i);
//            }
//        }
//
//        // Store result in output
//        for (int i = 0; i < headSize; i++) {
//            allXb.set(h * headSize + i, headOutput[i]);
//        }
//    }

//    /**
//     * Process a single attention head - TornadoVM compatible version
//     */
//    private static void processHeadTornado(
//            FloatArray allQ, FloatArray key_cache, FloatArray value_cache, FloatArray allXb,
//            int h, int headSize, int kvDim, int kvMul, long loff, int pos, FloatArray wrapAtt) {
//
//        // Local attention scores
////        float[] attScores = new float[pos + 1];
//
//        // Calculate attention scores for this head and all timesteps
//        for (int t = 0; t <= pos; t++) {
//            int kvHeadIdx = h / kvMul;
//            int keyOffset = (int) (loff + t * kvDim + kvHeadIdx * headSize);
//
//            float score = 0.0f;
//            for (int i = 0; i < headSize; i++) {
//                score += allQ.get(h * headSize + i) * key_cache.get(keyOffset + i);
//            }
//            score /= TornadoMath.sqrt(headSize);
//
//            wrapAtt.set(h * (pos + 1) + t, score);
////            attScores[t] = score;
//        }
//
//        // Softmax on attention scores
//        softmaxInPlace(wrapAtt, pos + 1);
//
//        // Initialize intermediate results
//        FloatArray localXb = new FloatArray(64); //TODO:
//        for (int i = 0; i < headSize; i++) {
//            localXb.set(i, 0.0f);
//        }
//
//        // Weighted sum of values
//        for (int t = 0; t <= pos; t++) {
//            int kvHeadIdx = h / kvMul;
//            int valueOffset = (int) (loff + t * kvDim + kvHeadIdx * headSize);
//
//            float a = wrapAtt.get(h * (pos + 1) + t);
////            float a = attScores[t];
//
//            for (int i = 0; i < headSize; i++) {
//                localXb.set(i, localXb.get(i) + a * value_cache.get(valueOffset + i));
//            }
//        }
//
//        // Store back to global xb
//        for (int i = 0; i < headSize; i++) {
//            allXb.set(h * headSize + i, localXb.get(i));
//        }
//    }
//
//    /**
//     * Softmax implementation that works in-place on a float array
//     */
//    private static void softmaxInPlace(FloatArray arr, int n) {
//        // Find max for numerical stability
//        float max = Float.NEGATIVE_INFINITY;
//        for (int i = 0; i < n; i++) {
//            if (arr.get(i) > max) {
//                max = arr.get(i);
//            }
//        }
//
//        // Compute exp and sum
//        float sum = 0.0f;
//        for (int i = 0; i < n; i++) {
//            arr.set(i, (float) TornadoMath.exp(arr.get(i) - max));
//            sum += arr.get(i);
//        }
//
//        // Normalize
//        for (int i = 0; i < n; i++) {
//            arr.set(i, arr.get(i) / sum);
//        }
//    }

    /**
     * Enhanced matrix multiplication for Q8 quantized weights with detailed debugging
     */
    public static void enhancedMatmulTornadoQ8(KernelContext context,
            ByteArray weights,
            FloatArray input,
            FloatArray output,
            int dim,
            FloatArray debugBuffer) {
        final int BLOCK_SIZE = 32; // Q8 block size
        final int BYTES_PER_BLOCK = 2 + BLOCK_SIZE; // 2 bytes for scale + 32 bytes for values

        int idx = context.globalIdx;

        // Debug info
        if (idx == 0) {
            debugBuffer.set(0, weights.getSize());   // Size of weights array
            debugBuffer.set(1, input.getSize());     // Size of input array
            debugBuffer.set(2, output.getSize());    // Size of output array
            debugBuffer.set(3, dim);                 // Dimension value

            // Record some input values for verification
            for (int i = 0; i < Math.min(5, input.getSize()); i++) {
                debugBuffer.set(5 + i, input.get(i));
            }
        }

        // Bounds check - crucial to avoid out-of-bounds access
        if (idx >= output.getSize()) {
            return;
        }

        float result = 0.0f;
        int thisOffset = idx * dim;

        // Compute the matrix multiplication with careful bounds checking
        for (int j = 0; j < dim; j++) {
            int index = thisOffset + j;

            // Calculate block position and ensure we don't exceed array bounds
            int blockIndex = index / BLOCK_SIZE;
            int withinBlockIndex = index % BLOCK_SIZE;
            int blockOffset = blockIndex * BYTES_PER_BLOCK;

            // Check if we have enough bytes left to read
            if (blockOffset + 1 >= weights.getSize()) {
                continue; // Skip this calculation if out of bounds
            }

            // Read scale (float16) for this block
            int scaleByte1 = weights.get(blockOffset) & 0xFF;
            int scaleByte2 = weights.get(blockOffset + 1) & 0xFF;
            short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
            float scale = decodeFloat16(scaleFloat16);

            // Check if we have enough bytes for the quantized value
            if (blockOffset + 2 + withinBlockIndex >= weights.getSize()) {
                continue; // Skip this calculation if out of bounds
            }

            // Read and dequantize the weight value
            byte quantized = weights.get(blockOffset + 2 + withinBlockIndex);

            // Check if input index is in bounds
            if (j >= input.getSize()) {
                continue; // Skip if input index is out of bounds
            }

            // Accumulate the result
            result += (quantized * scale) * input.get(j);
        }

        // Store the result
        output.set(idx, result);

        // Record the first few computed values for debugging
        if (idx < 5) {
            debugBuffer.set(idx, result);
        }
    }

    public static void checkLogitsValues(FloatArray logits, FloatArray debugBuffer) {
        int nonZeroCount = 0;
        float sum = 0.0f;
        float maxVal = Float.NEGATIVE_INFINITY;

        // Count non-zero values and compute basic stats
        for (int i = 0; i < Math.min(1000, logits.getSize()); i++) {
            float val = logits.get(i);
            sum += TornadoMath.abs(val);
            maxVal = TornadoMath.max(maxVal, val);

            if (TornadoMath.abs(val) > 1e-6f) {
                nonZeroCount++;
            }
        }

        // Store debugging info
        debugBuffer.set(0, nonZeroCount);
        debugBuffer.set(1, sum);
        debugBuffer.set(2, maxVal);
    }
}
