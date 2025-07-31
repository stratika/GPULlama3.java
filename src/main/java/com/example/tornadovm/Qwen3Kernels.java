package com.example.tornadovm;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
public class Qwen3Kernels {

    /**
     * For explicit copy out useful in debugging.
     * With this kernel we can store the values of an array to a tmp buffer at a timing of interest.
     * In the end of the taskgraph we copy out the tmp buffer to inspect the array values at the timing of interest.
     * @param srcBuffer the array we want to inspect.
     * @param dstBuffer the tmp buffer.
     */
    public static void dbgCopy(FloatArray srcBuffer, FloatArray dstBuffer) {
        for (@Parallel int i = 0; i < srcBuffer.getSize(); i++) {
            dstBuffer.set(i, srcBuffer.get(i));
        }
    }

    /**
     * RmsNorm with parallel offset:
     * The following 3 kernels implement rmsnorm in offset range in parallel for qCur and Kcur rmsnorm calculations.
     *
     * Step 1: Reduction.
     * This kernel implements rmsnorm in offset range in parallel for qCur and Kcur rmsnorm calculations.
     */
    public static void rmsnormReductionWithParallelOffset(KernelContext context, FloatArray output, FloatArray x, int localMemSize) {

        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupId = context.groupIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory with the provided size
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        // Load input value and compute square
        localX[lid] = x.get(gid);
        localX[lid] = localX[lid] * localX[lid];

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
            output.set(groupId, localX[0]);
        }
    }

    /**
     * RmsNorm with parallel offset:
     *
     * Step 2: Combines partial reduction outputs and computes final normalization.
     */
    public static void rmsnormFinalNormalizationWithParallelOffset(
            KernelContext context,
            FloatArray output, // size should be related to offsetIndex
            int offsetIndex,   // = config.numberOfHeads()
            int size,
            float ermsNorm) {

        int gid = context.globalIdx;

        // Only the index threads need to perform this calculation
        if (gid < offsetIndex) {
            // Combine partial sums from all workgroups
            float ss = output.get(gid);

            ss /= size;
            ss += ermsNorm;
            ss = 1.0f / TornadoMath.sqrt(ss);
            // in place
            output.set(gid, ss);  // Store the final scale factor
        }
    }

    /**
     * RmsNorm with parallel offset:
     *
     * Step 3: perform mapIndex operation.
     */
    public static void rmsnormMapIndexInPlaceWithParallelOffset(
            KernelContext context,
            FloatArray out,
            FloatArray weights,
            int size,
            FloatArray ss) {

        int gid = context.globalIdx;
        int groupId = context.groupIdx;

        float finalss = ss.get(groupId);

        if (gid < out.getSize()) { // TODO: check if redundant
            float a = weights.get(gid % size);
            float b = finalss * out.get(gid);
            out.set(gid, a * b);
        }
    }

    /**
     * RmsNorm with parallel offset:
     *
     * Optimized kernel that combines Step 1 (Reduction) and Step 2 (Normalization).
     */
    public static void rmsnormWithParallelOffset(
            KernelContext context,
            FloatArray output,
            FloatArray x,
            int localMemSize,
            int size,
            float ermsNorm) {

        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupId = context.groupIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory with the provided size
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        // Load input value and compute square
        localX[lid] = x.get(gid);
        localX[lid] = localX[lid] * localX[lid];

        // Perform parallel reduction within the work group
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        // Each workgroup performs the normalization
        if (lid == 0) {
            // Store the partial sum from each workgroup
            localX[0] /= size;
            localX[0] += ermsNorm;
            localX[0] = 1.0f / TornadoMath.sqrt(localX[0]);
            output.set(groupId, localX[0]);
        }
    }

    public static void ropeRotation(
            KernelContext context,
            IntArray position,
            FloatArray q,
            FloatArray k,
            int numberOfKeyValueHeads,
            int nEmbdHead) {

        int h = context.globalIdx;
        int ic = context.globalIdy;

        int rotn = h < numberOfKeyValueHeads ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        int poffset = h * nEmbdHead;
        int nComplEmbdHead = nEmbdHead / 2;

        // Compute RoPE frequencies for Qwen3
        float theta = 1000000.0f;
        int i = ic * 2; // match i in precompute (see RoPE.precomputeFreqsCis)
        float freq = 1.0f / TornadoMath.pow(theta, (float) i / (float) nEmbdHead);

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
            IntArray positionHolder,
            FloatArray wrapAtt,
            int layer, int contextLength) {

        int pos = positionHolder.get(0);
        int loff = layer * contextLength * nEmbdGqa;

        // Parallelize computation across attention heads
        for (@Parallel int h = 0; h < nHeads; h++) {
            // Process each head in parallel
            //noinspection ExternalInspection
            processHeadTornado(q, key_cache, value_cache, xb, h, nEmbdHead, /* headSize */
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

        // STEP 1: Calculate attention scores for all timesteps
        for (int t = 0; t <= pos; t++) {
            int kvHeadIdx = h / gqa;
            int keyOffset = (int) (loff + t * nEmbdGqa + kvHeadIdx * nEmbdHeadK); // line 255

            float score = 0.0f;
            for (int i = 0; i < nEmbdHeadK; i++) {
                score += allQ.get(h * nEmbdHeadK + i) * key_cache.get(keyOffset + i); // line 255
            }
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
        for (int i = 0; i < nEmbdHeadV; i++) {
            float weightedSum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                int kvHeadIdx = h / gqa;
                int valueOffset = (int) (loff + t * nEmbdGqa + kvHeadIdx * nEmbdHeadV); //line 273
                weightedSum += wrapAtt.get(headOffset + t) * value_cache.get(valueOffset + i);
            }
            allXb.set(h * nEmbdHeadV + i, weightedSum); // offset from line 266
        }
    }

}
// @formatter:on
