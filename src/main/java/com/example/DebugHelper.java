package com.example.debug;

import com.example.core.model.tensor.FloatTensor;
import com.example.inference.engine.impl.Llama;
import com.example.inference.engine.impl.Configuration;
import com.example.loader.weights.State;
import com.example.tornadovm.TornadoVMMasterPlan;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

/**
 * Utility class for debugging GPU vs CPU implementation differences
 */
public class DebugHelper {
    private static final boolean DEBUG_ENABLED = Boolean.parseBoolean(System.getProperty("llama.debug", "false"));
    private static final String LOG_FILE = "llama_debug.log";
    private static PrintWriter logWriter;

    static {
        if (DEBUG_ENABLED) {
            try {
                logWriter = new PrintWriter(new FileWriter(LOG_FILE, true));
                logWriter.println("\n==== New Debug Session Started ====");
                logWriter.flush();
            } catch (IOException e) {
                System.err.println("Failed to initialize debug log: " + e.getMessage());
            }
        }
    }

    /**
     * Compares CPU and GPU forward pass results and logs differences.
     * This version runs two separate passes rather than attempting to clone state.
     */
    public static void compareForwardImplementations(Llama model, int token, int position) {
        if (!DEBUG_ENABLED) return;

        log("Comparing forward implementations for token=" + token + ", position=" + position);

        // Create two separate states
        State cpuState = new State(model.configuration());
        State gpuState = new State(model.configuration());

        // Set initial token
        cpuState.latestToken = token;
        gpuState.latestToken = token;

        // Run CPU implementation
        log("Running CPU forward pass...");
        FloatTensor cpuLogits = Llama.forwardJava(model, cpuState, token, position);
        logTensorStats("CPU logits", cpuLogits);

        // Run GPU implementation
        log("Running GPU forward pass...");
        TornadoVMMasterPlan tornadoVMPlan = new TornadoVMMasterPlan(gpuState, model);
        FloatTensor gpuLogits = Llama.forwardTornadoVM(model, gpuState, token, position, tornadoVMPlan);
        logTensorStats("GPU logits", gpuLogits);

        // Compare and log
        compareLogits(cpuLogits, gpuLogits, model.tokenizer());

        // Clean up TornadoVM resources
        tornadoVMPlan.freeTornadoExecutionPlan();
    }

    /**
     * Debug a specific layer by comparing CPU and GPU outputs for just that layer
     */
    public static void debugLayer(Llama model, int token, int position, int layerIndex) {
        if (!DEBUG_ENABLED) return;

        log("Debugging layer " + layerIndex + " for token=" + token + ", position=" + position);

        Configuration config = model.configuration();
        if (layerIndex >= config.numberOfLayers) {
            log("ERROR: Layer index " + layerIndex + " is out of bounds (model has " +
                    config.numberOfLayers + " layers)");
            return;
        }

        // Create separate states for CPU and GPU
        State cpuState = new State(config);
        State gpuState = new State(config);

        // Prepare for layer execution
        cpuState.latestToken = token;
        gpuState.latestToken = token;

        // Load the token embedding into both states
        model.weights().token_embedding_table.copyTo(token * config.dim, cpuState.x, 0, config.dim);
        model.weights().token_embedding_table.copyTo(token * config.dim, gpuState.x, 0, config.dim);

        log("Embeddings loaded. Running forward pass up to layer " + layerIndex);

        // Process all previous layers using CPU implementation to ensure consistent input
        for (int l = 0; l < layerIndex; l++) {
            log("Pre-processing layer " + l + " with CPU implementation");
            processLayerCPU(model, cpuState, position, l);

            // Copy state from CPU to GPU for the next layer
            if (l == layerIndex - 1) {
                copyTensorState(cpuState, gpuState);
                log("State copied from CPU to GPU after layer " + l);
                logTensorStats("CPU state X before layer " + layerIndex, cpuState.x);
                logTensorStats("GPU state X before layer " + layerIndex, gpuState.x);
            }
        }

        // Now run the target layer with both implementations and compare
        log("Running layer " + layerIndex + " with CPU implementation");
        processLayerCPU(model, cpuState, position, layerIndex);
        logTensorStats("CPU state X after layer " + layerIndex, cpuState.x);

        log("Running layer " + layerIndex + " with GPU implementation");
        // For GPU implementation, we'd need to modify the code to run a single layer
        // with TornadoVM. This would require changes to the TornadoVMMasterPlan class.
        log("WARNING: Individual layer execution for GPU is not yet implemented");

        log("Layer debugging complete");
    }

    /**
     * Process a single layer using CPU implementation
     */
    private static void processLayerCPU(Llama model, State state, int position, int layer) {
        Configuration config = model.configuration();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads;
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        // attention rmsnorm
        Llama.rmsnorm(state.xb, state.x, model.weights().rms_att_weight[layer], dim, config.rmsNormEps);

        // qkv matmuls for this position
        model.weights().wq[layer].matmul(state.xb, state.q, dim, dim);
        model.weights().wk[layer].matmul(state.xb, state.k, kvDim, dim);
        model.weights().wv[layer].matmul(state.xb, state.v, kvDim, dim);

        // RoPE relative positional encoding
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % headSize;
            float fcr = model.weights().freq_cis_real.get(position * (headSize / 2) + (head_dim / 2));
            float fci = model.weights().freq_cis_imag.get(position * (headSize / 2) + (head_dim / 2));
            int rotn = i < kvDim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                FloatTensor vec = v == 0 ? state.q : state.k;
                float v0 = vec.getFloat(i);
                float v1 = vec.getFloat(i + 1);
                vec.setFloat(i, v0 * fcr - v1 * fci);
                vec.setFloat(i + 1, v0 * fci + v1 * fcr);
            }
        }

        // save key,value to cache
        state.k.copyTo(0, state.keyCache[layer], position * kvDim, kvDim);
        state.v.copyTo(0, state.valueCache[layer], position * kvDim, kvDim);

        // Process attention for each head
        for (int h = 0; h < config.numberOfHeads; h++) {
            int qOffset = h * headSize;
            int attOffset = h * config.contextLength;

            // Calculate attention scores
            for (int t = 0; t <= position; t++) {
                int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                float score = state.q.dot(qOffset, state.keyCache[layer], keyCacheOffset, headSize);
                score /= sqrtHeadSize;
                state.att.setFloat(attOffset + t, score);
            }

            // Apply softmax
            state.att.softmaxInPlace(attOffset, position + 1);

            // Weighted sum of values
            int xbOffset = h * headSize;
            state.xb.fillInPlace(xbOffset, headSize, 0f);

            for (int t = 0; t <= position; t++) {
                int vOffset = t * kvDim + (h / kvMul) * headSize;
                float a = state.att.getFloat(attOffset + t);
                state.xb.saxpyInPlace(xbOffset, state.valueCache[layer], vOffset, headSize, a);
            }
        }

        // Final attention output
        model.weights().wo[layer].matmul(state.xb, state.xb2, dim, dim);
        state.x.addInPlace(state.xb2);

        // FFN
        Llama.rmsnorm(state.xb, state.x, model.weights().rms_ffn_weight[layer], dim, config.rmsNormEps);
        model.weights().w1[layer].matmul(state.xb, state.hb, config.hiddenDim, dim);
        model.weights().w3[layer].matmul(state.xb, state.hb2, config.hiddenDim, dim);

        // Apply SiLU activation
        state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
        state.hb.multiplyInPlace(state.hb2);

        // Final FFN output
        model.weights().w2[layer].matmul(state.hb, state.xb, dim, config.hiddenDim);
        state.x.addInPlace(state.xb);
    }

    /**
     * Copy essential state tensors from source to target
     */
    private static void copyTensorState(State source, State target) {
        copyTensor(source.x, target.x);
        copyTensor(source.xb, target.xb);
        copyTensor(source.xb2, target.xb2);
        copyTensor(source.q, target.q);
        copyTensor(source.k, target.k);
        copyTensor(source.v, target.v);
        copyTensor(source.hb, target.hb);
        copyTensor(source.hb2, target.hb2);
        copyTensor(source.att, target.att);

        // Copy KV cache for all layers
        for (int l = 0; l < source.keyCache.length; l++) {
            if (l < target.keyCache.length) {
                copyTensor(source.keyCache[l], target.keyCache[l]);
                copyTensor(source.valueCache[l], target.valueCache[l]);
            }
        }

        // Also transfer to TornadoVM buffers
        for (int i = 0; i < Math.min(source.x.size(), target.wrapX.getSize()); i++) {
            target.wrapX.set(i, source.x.getFloat(i));
        }
    }

    private static void copyTensor(FloatTensor source, FloatTensor target) {
        if (source == null || target == null) return;
        for (int i = 0; i < Math.min(source.size(), target.size()); i++) {
            target.setFloat(i, source.getFloat(i));
        }
    }

    /**
     * Log tensor statistics
     */
    private static void logTensorStats(String name, FloatTensor tensor) {
        if (!DEBUG_ENABLED || logWriter == null || tensor == null) return;

        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        float sum = 0;
        float absSum = 0;
        int nanCount = 0;
        int infCount = 0;

        for (int i = 0; i < tensor.size(); i++) {
            float val = tensor.getFloat(i);

            if (Float.isNaN(val)) {
                nanCount++;
                continue;
            }

            if (Float.isInfinite(val)) {
                infCount++;
                continue;
            }

            min = Math.min(min, val);
            max = Math.max(max, val);
            sum += val;
            absSum += Math.abs(val);
        }

        float mean = sum / tensor.size();
        float absAvg = absSum / tensor.size();

        log(name + " stats:");
        log("  Size: " + tensor.size());
        log("  Range: [" + min + ", " + max + "]");
        log("  Mean: " + mean);
        log("  Average absolute value: " + absAvg);
        log("  NaN count: " + nanCount);
        log("  Infinity count: " + infCount);

        // Print sample values
        if (tensor.size() > 0) {
            StringBuilder sb = new StringBuilder("  Sample values: ");
            for (int i = 0; i < Math.min(10, tensor.size()); i++) {
                sb.append(tensor.getFloat(i)).append(" ");
            }
            log(sb.toString());
        }
    }

    /**
     * Compare CPU and GPU logits
     */
    private static void compareLogits(FloatTensor cpuLogits, FloatTensor gpuLogits,
            com.example.tokenizer.impl.Tokenizer tokenizer) {
        if (!DEBUG_ENABLED || logWriter == null) return;
        if (cpuLogits == null || gpuLogits == null) {
            log("ERROR: Cannot compare logits - one is null");
            return;
        }

        if (cpuLogits.size() != gpuLogits.size()) {
            log("ERROR: Logits size mismatch - CPU: " + cpuLogits.size() + ", GPU: " + gpuLogits.size());
            return;
        }

        float maxDiff = 0;
        int maxDiffIdx = -1;
        float sumDiff = 0;
        int largeDiscrepancies = 0;

        for (int i = 0; i < cpuLogits.size(); i++) {
            float cpuVal = cpuLogits.getFloat(i);
            float gpuVal = gpuLogits.getFloat(i);
            float diff = Math.abs(cpuVal - gpuVal);
            sumDiff += diff;

            if (diff > maxDiff) {
                maxDiff = diff;
                maxDiffIdx = i;
            }

            if (diff > 1.0) {
                largeDiscrepancies++;
            }
        }

        float avgDiff = sumDiff / cpuLogits.size();

        log("Logits comparison:");
        log("  Average difference: " + avgDiff);
        log("  Maximum difference: " + maxDiff + " at index " + maxDiffIdx);
        log("  Number of large discrepancies (diff > 1.0): " + largeDiscrepancies);

        // Get top CPU tokens
        int[] cpuTopTokens = getTopTokenIndices(cpuLogits, 5);
        log("CPU top tokens:");
        for (int i = 0; i < cpuTopTokens.length; i++) {
            int tokenId = cpuTopTokens[i];
            float prob = cpuLogits.getFloat(tokenId);
            String tokenStr = tokenizer.decode(Arrays.asList(tokenId));
            log(String.format("  %d. Token %d (%s) - logit: %.4f",
                    i+1, tokenId, tokenStr, prob));
        }

        // Get top GPU tokens
        int[] gpuTopTokens = getTopTokenIndices(gpuLogits, 5);
        log("GPU top tokens:");
        for (int i = 0; i < gpuTopTokens.length; i++) {
            int tokenId = gpuTopTokens[i];
            float prob = gpuLogits.getFloat(tokenId);
            String tokenStr = tokenizer.decode(Arrays.asList(tokenId));
            log(String.format("  %d. Token %d (%s) - logit: %.4f",
                    i+1, tokenId, tokenStr, prob));
        }
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

    static void log(String message) {
        if (logWriter != null) {
            logWriter.println(message);
            logWriter.flush();
        }
        System.err.println("[DEBUG] " + message);
    }

    /**
     * Close resources on shutdown
     */
    public static void shutdown() {
        if (logWriter != null) {
            logWriter.close();
        }
    }
}