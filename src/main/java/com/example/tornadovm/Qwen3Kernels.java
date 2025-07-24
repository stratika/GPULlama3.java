package com.example.tornadovm;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

public class Qwen3Kernels {

    //public static void dbgCopy(FloatArray destKeyCache, FloatArray srcKey, FloatArray destValueCache, FloatArray srcValue, IntArray positioNlayer, int kvDim, int layer, int contextLength) {
    public static void dbgCopy(FloatArray srcBuffer, FloatArray dstBuffer, IntArray positioNlayer, int layer) {
        //int position = positioNlayer.get(0);
        //if (position == 1) {
            for (@Parallel int i = 0; i < srcBuffer.getSize(); i++) {
                dstBuffer.set(i, srcBuffer.get(i));
            }
        //}
    }

    public static void reductionOneBlockWithLayerWithOffset(
            KernelContext context,
            FloatArray output,
            FloatArray x,
            int offset,
            int size,
            float ermsNorm,
            int localMemSize) {

        int gid = context.globalIdx; // 0 - nEmbHead = 128
        int lid = context.localIdx;  // 0 - state.localsize [
        int groupId = context.groupIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory with the provided size
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        // Load input value and compute square
        int globalReadIndex = gid + offset;
        if (gid < size && globalReadIndex < x.getSize()) {
            localX[lid] = x.get(globalReadIndex);
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

//        // Only the first thread in the first workgroup computes the final normalization factor
//        if (gid == 0) {
//            // Combine partial sums from all workgroups
//            float ss = 0.0f;
//            for (int i = 1; i <= (size / localMemSize); i++) {  // Assuming 8 workgroups
//                ss += output.get(i);
//            }
//
//            ss /= size;
//            ss += ermsNorm;
//            ss = 1.0f / TornadoMath.sqrt(ss);
//            output.set(0, ss);  // Store the final scale factor
//        }
    }

    /**
     * Normalize and scale (in-place) of rmsnorm operation.
     */
    public static void mapIndexInPlace(KernelContext context, FloatArray out, /*FloatArray x,*/ FloatArray weights, int offset, int size, FloatArray ss) {
        int gid = context.globalIdx; // 0 - size
        int index = offset + gid;

        float finalss = ss.get(0);
        //out.set(index, weights.get(index % size) * (finalss * x.get(index)));
        //out.set(index, weights.get(index) * (finalss * x.get(index)));
        //if (index < offset + size) {
        if (index < out.getSize()) { // TODO: check if redundant
            float a = weights.get(index % size);
            float b = finalss * out.get(index);
            out.set(index, a * b);
        }

        context.globalBarrier();
        // reset ss
        if (gid < ss.getSize()) {
            ss.set(gid, 0.0f);
        }
    }

    public static void ropeRotation(KernelContext context,
            IntArray position,
            FloatArray q,
            FloatArray k,
            int numberOfKeyValueHeads,
            int nEmbdHead) {
        //System.out.println("ropeRotationSplit");
        int h = context.globalIdx;
        int ic = context.globalIdy;

        int rotn = h < numberOfKeyValueHeads ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        int poffset = h * nEmbdHead;
        int nComplEmbdHead = nEmbdHead / 2;

        // Compute RoPE frequencies for Qwen3
        //float freq = 1.0f / TornadoMath.pow(10000.0f, (2.0f * ic) / (float) nEmbdHead);
        float theta = 1000000.0f;
        int i = ic * 2; // match i in precompute (see RoPE.precomputeFreqsCis)
        float freq = 1.0f / TornadoMath.pow(theta, (float)i / (float)nEmbdHead);

        float val = position.get(0) * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        float v0q = q.get(poffset + ic);
        float v1q = q.get(poffset + ic + nComplEmbdHead);
        q.set(poffset + ic, v0q * fcr - v1q * fci);
        q.set(poffset + ic + nComplEmbdHead, v0q * fci + v1q * fcr);

        if (rotn > 1 && (poffset + ic + nComplEmbdHead) < k.getSize()) {
            float v0k = k.get(poffset + ic);
            float v1k = k.get(poffset + ic + nComplEmbdHead);
            k.set(poffset + ic, v0k * fcr - v1k * fci);
            k.set(poffset + ic + nComplEmbdHead, v0k * fci + v1k * fcr);
        }

    }

    public static void processHeadsParallel(
            FloatArray q,
            FloatArray key_cache,
            FloatArray value_cache,
            FloatArray xb,
            int nHeads,
            int nEmbdHead, /* = nEmbdHead, replace headSize in lines: 244, 253,  */
            int nEmbdHeadK, /* = config.numberOfHeadsKey(), replace headSize in line 255 */
            int nEmbdHeadV, /* = config.numberOfHeadsValue(), replace headSize in lines: 266, 268, 273 */
            int nEmbdGqa, /* kvDim */
            int gqa, /* kvMul */
            int seqLen,
            IntArray positionHolder,
            FloatArray wrapAtt,
            int layer, int contextLength) {

        int pos = positionHolder.get(0);
        //int loff = layer * contextLength * kvDim;
        int loff = layer * contextLength * nEmbdGqa;

        // Parallelize computation across attention heads
        for (@Parallel int h = 0; h < nHeads; h++) {
            // Process each head in parallel
            //noinspection ExternalInspection
            processHeadTornado(q, key_cache, value_cache, xb,
                    h,
                    nEmbdHead, /* headSize */
                    nEmbdHeadK, /* headSize in line 255 */
                    nEmbdHeadV, /* headSize in lines: 266, 268, 273 */
                    nEmbdGqa, /* kvDim */
                    gqa, /* kvMul */
                    loff, pos, wrapAtt, contextLength);
        }
    }

    private static void processHeadTornado(
            FloatArray allQ,
            FloatArray key_cache,
            FloatArray value_cache,
            FloatArray allXb,
            int h,
            int nEmbdHead, /* = nEmbdHeadV, replace headSize in lines: 244, 253,  */
            int nEmbdHeadK, /* = config.numberOfHeadsKey(), replace headSize in line 255 */
            int nEmbdHeadV, /* = config.numberOfHeadsValue(), replace headSize in lines: 266, 268, 273 */
            int nEmbdGqa, /* kvDim */
            int gqa, /* kvMul */
            long loff,
            int pos,
            FloatArray wrapAtt,
            int contextLength) {

        // Base index for this head's attention weights
        int headOffset = h * (pos + 1);
        //int headOffset = h * contextLength;

        // STEP 1: Calculate attention scores for all timesteps
        for (int t = 0; t <= pos; t++) {
            //int kvHeadIdx = h / kvMul;
            int kvHeadIdx = h / gqa;
            //int keyOffset = (int) (loff + t * kvDim + kvHeadIdx * headSize);
            int keyOffset = (int) (loff + t * nEmbdGqa + kvHeadIdx * nEmbdHeadK); // line 255

            float score = 0.0f;
            //for (int i = 0; i < headSize; i++) {
            for (int i = 0; i < nEmbdHeadK; i++) {
                //score += allQ.get(h * headSize + i) * key_cache.get(keyOffset + i);
                score += allQ.get(h * nEmbdHeadK + i) * key_cache.get(keyOffset + i); // line 255
            }
            //score = score / TornadoMath.sqrt(headSize);
            score = score / TornadoMath.sqrt(nEmbdHead); // line 257

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
        //for (int i = 0; i < headSize; i++) {
        for (int i = 0; i < nEmbdHeadV; i++) {
            float weightedSum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                //int kvHeadIdx = h / kvMul;
                int kvHeadIdx = h / gqa;
                //int valueOffset = (int) (loff + t * kvDim + kvHeadIdx * headSize);
                int valueOffset = (int) (loff + t * nEmbdGqa + kvHeadIdx * nEmbdHeadV); //line 273
                weightedSum += wrapAtt.get(headOffset + t) * value_cache.get(valueOffset + i);
            }
            //allXb.set(h * headSize + i, weightedSum);
            allXb.set(h * nEmbdHeadV + i, weightedSum); // offset from line 266
        }
    }

    public static void matrixVectorGenericWithResidual(
            KernelContext context,
            FloatArray v,           // vector = [2048]
            FloatArray out,         // out    = [1024]
            HalfFloatArray m,       // matrix = [2048, 1024]
            int dim1,               // dim1   = 2048, vectorSize
            int dim0,               // dim0   = 1024, outputSize
            int localWorkGroupSize) {

        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= dim0) {
            return;
        }

        float sum = matrixVectorRowMajorOptimized(context, localSize, v, m, dim1, dim0);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float result = out.get(rowId) + sum;
            out.set(rowId, result);
        }
    }

    public static float matrixVectorRowMajorOptimized(
            KernelContext context,
            int localSize,
            FloatArray v,
            HalfFloatArray m,
            int dim1,
            int dim0
    ) {
        int rowId = context.groupIdx; // 0-dim
        int localId = context.localIdx; // 0-32

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localSize);

        int rowOffset = rowId * dim1;

        // Each thread calculates partial dot product
        float partialSum = 0.0f;
        for (int j = localId; j < dim1; j += localSize) {
            int matrixIdx = rowOffset + j;
            partialSum += m.get(matrixIdx).getFloat32() * v.get(j);
            //partialSum += w.get(rowOffset + j).getFloat32() * x.get(j);
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

    public static float matrixVectorRowMajorOptimized2(
            KernelContext context,
            int localSize,
            FloatArray v,           // input vector [2048]
            HalfFloatArray m,       // matrix [2048, 1024]
            int vectorSize,         // 2048
            int outputSize,
            int rowId               // which output row we're computing (0-1023)
    ) {
        int localId = context.localIdx; // 0 to localSize-1

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localSize);

        // For matrix [2048, 1024], if we want row 'rowId' of the OUTPUT,
        // we need to compute dot product of INPUT vector with COLUMN 'rowId' of the matrix
        // Matrix element [i][j] is at index i * outputSize + j
        // We want column 'rowId', so elements are at: 0*outputSize + rowId, 1*outputSize + rowId, etc.

        // Each thread calculates partial dot product
        float partialSum = 0.0f;
        for (int i = localId; i < vectorSize; i += localSize) {
            int matrixIdx = i * outputSize + rowId;  // Column-wise access for row rowId
            partialSum += m.get(matrixIdx).getFloat32() * v.get(i);
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

}
