package com.example;

import com.example.aux.Parallel;
import com.example.core.model.tensor.FloatTensor;
import com.example.debug.DebugHelper;
import com.example.inference.engine.impl.Configuration;
import com.example.inference.engine.impl.Llama;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.tornadovm.TornadoVMMasterPlan;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

/**
 * Debug utility for comparing GPU and CPU forward pass in Llama model
 */
public class LlamaForwardDebug {
    private static final boolean DETAILED_DEBUG = Boolean.parseBoolean(System.getProperty("llama.DetailedDebug", "false"));
    private static PrintWriter debugWriter;

    static {
        try {
            if (DETAILED_DEBUG) {
                debugWriter = new PrintWriter(new FileWriter("llama_forward_debug.log"));
                debugWriter.println("=== Llama Forward Pass Debug Log ===");
                debugWriter.flush();
            }
        } catch (IOException e) {
            System.err.println("Failed to initialize debug log: " + e.getMessage());
        }
    }

    /**
     * Debug-enabled forward pass on CPU that logs intermediate states
     */
    public static FloatTensor debugForwardJava(Llama model, State state, int token, int position) {
        // a few convenience variables
        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        logDebug("CPU forward pass - token: " + token + ", position: " + position);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);
        logTensorState("x after embedding", state.x);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers; l++) {
            logDebug("CPU Layer " + l);

            // attention rmsnorm
            Llama.rmsnorm(state.xb, state.x, weights.rms_att_weight[l], dim, config.rmsNormEps);
            logTensorState("xb after rmsnorm", state.xb);

            // qkv matmuls for this position
            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            logTensorState("q after matmul", state.q);
            logTensorState("k after matmul", state.k);
            logTensorState("v after matmul", state.v);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % headSize;
                float fcr = weights.freq_cis_real.get(position * (headSize / 2) + (head_dim / 2));
                float fci = weights.freq_cis_imag.get(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k; // the vector to rotate (query or key)
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }

            logTensorState("q after RoPE", state.q);
            logTensorState("k after RoPE", state.k);

            // save key,value at this time step (position) to our kv cache
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            // multihead attention. iterate over all heads
            Parallel.parallelFor(0, config.numberOfHeads, h -> {
                // get the query vector for this head
                int qOffset = h * headSize;

                // attention scores for this head
                int attOffset = h * config.contextLength;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position; t++) {
                    // get the key vector for this head and at this timestep
                    int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att.setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att.softmaxInPlace(attOffset, position + 1);

                // weighted sum of the values, store back into xb
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    // get the value vector for this head and at this timestep
                    int vOffset = t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att.getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            logTensorState("xb after attention", state.xb);

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);
            logTensorState("xb2 after wo matmul", state.xb2);

            // residual connection back into x
            state.x.addInPlace(state.xb2);
            logTensorState("x after residual", state.x);

            // ffn rmsnorm
            Llama.rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], dim, config.rmsNormEps);
            logTensorState("xb after ffn rmsnorm", state.xb);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim, dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim, dim);

            logTensorState("hb after w1 matmul", state.hb);
            logTensorState("hb2 after w3 matmul", state.hb2);

            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
            logTensorState("hb after silu", state.hb);

            // elementwise multiply with w3(x)
            state.hb.multiplyInPlace(state.hb2);
            logTensorState("hb after multiply with hb2", state.hb);

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim);
            logTensorState("xb after w2 matmul", state.xb);

            // residual connection
            state.x.addInPlace(state.xb);
            logTensorState("x after ffn residual", state.x);
        }

        Llama.rmsnorm(state.x, state.x, weights.rms_final_weight, dim, config.rmsNormEps);
        logTensorState("x after final rmsnorm", state.x);

        weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim);
        logTensorState("logits", state.logits);

        return state.logits;
    }

    /**
     * Debug-enabled forward pass on GPU that logs intermediate states
     */
    public static FloatTensor debugForwardTornadoVM(Llama model, State state, int token, int position,
            TornadoVMMasterPlan tornadoVMMasterPlan) {
        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;

        logDebug("GPU forward pass - token: " + token + ", position: " + position);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);
        logTensorState("GPU x after embedding", state.x);

        // Copy to TornadoVM buffer
        System.arraycopy(state.x.asMemorySegment().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT), 0,
                state.wrapX.toHeapArray(), 0, dim);

        // Run TornadoVM forward pass
        tornadoVMMasterPlan.tornadoVMForwardExecute(position);

        // Log TornadoVM state after execution
        logTensorState("GPU logits", state.logits);

        return state.logits;
    }

    /**
     * Log tensor state with optional stats
     */
    private static void logTensorState(String name, FloatTensor tensor) {
        if (!DETAILED_DEBUG || debugWriter == null) return;

        // Calculate basic stats
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        float sum = 0;
        float sumSq = 0;

        for (int i = 0; i < Math.min(tensor.size(), 1000); i++) {
            float val = tensor.getFloat(i);
            min = Math.min(min, val);
            max = Math.max(max, val);
            sum += val;
            sumSq += val * val;
        }

        float mean = sum / Math.min(tensor.size(), 1000);
        float stddev = (float) Math.sqrt(sumSq / Math.min(tensor.size(), 1000) - mean * mean);

        debugWriter.println(name + ": size=" + tensor.size() +
                ", min=" + min + ", max=" + max +
                ", mean=" + mean + ", stddev=" + stddev);

        // Print first few values
        if (tensor.size() > 0) {
            debugWriter.print("  First 10 values: ");
            for (int i = 0; i < Math.min(10, tensor.size()); i++) {
                debugWriter.print(tensor.getFloat(i) + " ");
            }
            debugWriter.println();
        }

        debugWriter.flush();
    }

    private static void logDebug(String message) {
        if (!DETAILED_DEBUG || debugWriter == null) return;
        debugWriter.println(message);
        debugWriter.flush();
    }

    /**
     * Compare CPU and GPU logits
     */
    public static void compareLogits(FloatTensor cpuLogits, FloatTensor gpuLogits) {
        if (!DETAILED_DEBUG || debugWriter == null) return;
        if (cpuLogits == null || gpuLogits == null) {
            logDebug("ERROR: Cannot compare logits - one is null");
            return;
        }

        if (cpuLogits.size() != gpuLogits.size()) {
            logDebug("ERROR: Logits size mismatch - CPU: " + cpuLogits.size() + ", GPU: " + gpuLogits.size());
            return;
        }

        float maxDiff = 0;
        int maxDiffIdx = -1;
        float sumDiff = 0;

        for (int i = 0; i < cpuLogits.size(); i++) {
            float cpuVal = cpuLogits.getFloat(i);
            float gpuVal = gpuLogits.getFloat(i);
            float diff = Math.abs(cpuVal - gpuVal);
            sumDiff += diff;

            if (diff > maxDiff) {
                maxDiff = diff;
                maxDiffIdx = i;
            }
        }

        float avgDiff = sumDiff / cpuLogits.size();

        logDebug("Logits comparison: maxDiff=" + maxDiff + " at idx=" + maxDiffIdx +
                ", avgDiff=" + avgDiff);

        // Print top tokens from each
        int[] cpuTopTokens = getTopTokenIndices(cpuLogits, 5);
        int[] gpuTopTokens = getTopTokenIndices(gpuLogits, 5);

        logDebug("CPU top token indices: " + Arrays.toString(cpuTopTokens));
        logDebug("GPU top token indices: " + Arrays.toString(gpuTopTokens));
    }

    /**
     * Get indices of top N values in tensor
     */
    private static int[] getTopTokenIndices(FloatTensor tensor, int n) {
        int[] indices = new int[n];
        float[] values = new float[n];

        // Initialize with minimum values
        for (int i = 0; i < n; i++) {
            indices[i] = -1;
            values[i] = Float.NEGATIVE_INFINITY;
        }

        // Find top N values
        for (int i = 0; i < tensor.size(); i++) {
            float val = tensor.getFloat(i);

            // Check if this value belongs in the top N
            for (int j = 0; j < n; j++) {
                if (val > values[j]) {
                    // Shift everything down
                    for (int k = n - 1; k > j; k--) {
                        values[k] = values[k - 1];
                        indices[k] = indices[k - 1];
                    }

                    // Insert the new value
                    values[j] = val;
                    indices[j] = i;
                    break;
                }
            }
        }

        return indices;
    }

    /**
     * Release resources
     */
    public static void shutdown() {
        if (debugWriter != null) {
            debugWriter.close();
        }
    }
}