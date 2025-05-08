package com.example.inference.engine.impl;

import com.example.aux.Parallel;
import com.example.core.model.tensor.FloatTensor;
import com.example.inference.Sampler;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tornadovm.TornadoVMCompute;
import com.example.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.lang.foreign.MemorySegment;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public record Llama(Configuration configuration, Tokenizer tokenizer, Weights weights) {
        private static final int BATCH_SIZE = Integer.getInteger("llama.BatchSize", 16);

    public static void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        // calculate sum of squares
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }

    public static FloatTensor forwardJava(Llama model, State state, int token, int position) {
        // a few convenience variables
        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers; l++) {
            // attention rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], dim, config.rmsNormEps);

            // qkv matmuls for this position

            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

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

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim;
            // kv cache layer offset for convenience
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            // multihead attention. iterate over all heads
            Parallel.parallelFor(0, config.numberOfHeads, h -> {
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;

                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att.setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att.softmaxInPlace(attOffset, position + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att.getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection back into x
            state.x.addInPlace(state.xb2);

            // ffn rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], dim, config.rmsNormEps);

            //            System.out.println("x " + weights.w1.toString() + " " + weights.w2.toString() + " " + weights.w3.toString());
            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim, dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim, dim);

            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            // elementwise multiply with w3(x)
            state.hb.multiplyInPlace(state.hb2);

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim);

            // residual connection
            state.x.addInPlace(state.xb);
        }

        rmsnorm(state.x, state.x, weights.rms_final_weight, dim, config.rmsNormEps);

        weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim);

        return state.logits;
    }

    /**
     * Performs the initial embedding lookup and triggers the TornadoVM accelerated forward pass for an LLM token.
     *
     * <p>This method handles the first phase of processing a token through the transformer model:
     * <ol>
     *   <li>Copies the token embedding from the model's embedding table to the state's buffer</li>
     *   <li>Delegates the transformer layer processing to TornadoVM through the master plan</li>
     * </ol>
     *
     * <p>The token embedding lookup happens on the CPU using {@link MemorySegment} operations,
     * while the subsequent transformer layers processing is offloaded to the accelerator through
     * TornadoVM for improved performance.
     *
     * @param model
     *         The Llama model containing weights and configuration parameters
     * @param state
     *         The current execution state holding input/output tensors and temporary buffers
     * @param token
     *         The input token ID to process
     * @param position
     *         The position of this token in the sequence context window
     * @param tornadoVMMasterPlan
     *         The execution plan for TornadoVM acceleration
     * @return FloatTensor containing the output logits for token prediction
     */
    public static FloatArray forwardTornadoVM( //
            Llama model,  //
            State state,  //
            int token,    //
            int position,   //
            TornadoVMMasterPlan tornadoVMMasterPlan) { //

        MemorySegment.copy(model.weights.tokenEmbeddingTable.getSegment(), token * model.configuration.dim * Float.BYTES, state.wrapX.getSegment(), 0, model.configuration.dim * Float.BYTES);

        return tornadoVMMasterPlan.tornadoVMForwardExecute(position);
    }
    public static List<Integer> generateTokensGPU(Llama model, State state, int startPosition, List<Integer> promptTokens,
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo) {
        // === GPU-Specific Optimizations ===

        // 1. Pre-allocate the TornadoVM plan just once
        TornadoVMMasterPlan tornadoVMPlan = new TornadoVMMasterPlan(state, model);

        // 2. Perform warmup with extra iterations to ensure JIT compilation is complete
        tornadoVMPlan.executionPlan.withWarmUp(); // Increased warmup iterations
        tornadoVMPlan.forceCopyInReadOnlyData();  // Force copy-in read-only weights

        // 3. Pre-upload prompt tokens to device memory if possible
        // Assuming you have a method to upload tokens to GPU memory
        if (promptTokens.size() > 0) {
//            tornadoVMPlan.preloadPromptTokens(promptTokens);
        }

        // === Setup and Initialization ===
        long startNanos = System.nanoTime();
        long inferenceStartNanos = 0;

        // Pre-validate the max tokens to avoid checking in the loop
        int actualMaxTokens = Math.min(maxTokens > 0 ? maxTokens : model.configuration().contextLength,
                model.configuration().contextLength);

        // Preallocate with expected capacity to avoid resizing
        List<Integer> generatedTokens = new ArrayList<>(
                Math.min(256, actualMaxTokens - promptTokens.size())); // Conservative estimate

        // === Token Generation Loop ===
        int currentToken = state.latestToken;
        int nextToken;
        int promptIndex = 0;
        int pos = startPosition;

        // Use more efficient direct array access for prompt tokens if possible
        int[] promptTokenArray = null;
        if (promptTokens instanceof ArrayList) {
            // Try to extract the underlying array for faster access
            try {
                // This is a performance optimization that may not work on all JVMs
                // Fall back to regular list access if it fails
                promptTokenArray = promptTokens.stream().mapToInt(Integer::intValue).toArray();
            } catch (Exception e) {
                // Fall back to list access
            }
        }

        // Main generation loop
        while (pos < actualMaxTokens) {
            // GPU Forward Pass - No conditional check since we know we're using GPU
            FloatArray logits = forwardTornadoVM(model, state, currentToken, pos, tornadoVMPlan);

            // Process prompt tokens if still remaining
            if (promptIndex < promptTokens.size()) {
                // Get next prompt token (using array access if available)
                nextToken = promptTokenArray != null ?
                        promptTokenArray[promptIndex++] :
                        promptTokens.get(promptIndex++);

                if (echo) {
                    // Decode and output token
                    System.err.print(Tokenizer.replaceControlCharacters(
                            model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                // Mark first inference token
                if (inferenceStartNanos == 0) {
                    inferenceStartNanos = System.nanoTime();
                }

                // Sample next token - use GPU sampling if available
                nextToken = sampler.sampleToken(logits);

                // Output if needed
                if (echo) {
                    System.err.print(Tokenizer.replaceControlCharacters(
                            model.tokenizer().decode(List.of(nextToken))));
                }

                // Store token
                generatedTokens.add(nextToken);

                // Check stop condition
                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }

            // Update for next iteration
            currentToken = nextToken;
            state.latestToken = currentToken;
            pos++;
        }

        // === Performance Metrics ===
        long endNanos = System.nanoTime();
        double totalSeconds = (endNanos - startNanos) / 1_000_000_000.0;
        int totalTokens = promptIndex + generatedTokens.size();

        // Calculate and print metrics
        double tokensPerSecond = totalTokens / totalSeconds;

        // Print performance metrics like the C implementation
        System.err.printf("\n\nachieved tok/s: %.2f. Tokens: %d, seconds: %.2f\n",
                tokensPerSecond, totalTokens, totalSeconds);

        // Add extra GPU performance metrics
//        if (inferenceStartNanos > 0) {
//            double inferenceSeconds = (endNanos - inferenceStartNanos) / 1_000_000_000.0;
//            System.err.printf("GPU generation speed: %.2f tok/s\n",
//                    generatedTokens.size() / inferenceSeconds);
//        }

        // Release GPU resources
        tornadoVMPlan.freeTornadoExecutionPlan();

        return generatedTokens;
    }

    public static List<Integer> generateTokens(Llama model, State state, int startPosition, List<Integer> promptTokens,
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        // Initialize TornadoVM plan if enabled
        TornadoVMMasterPlan tornadoVMPlan = null;
        if (TornadoVMCompute.TORNADOVM) {
            tornadoVMPlan = new TornadoVMMasterPlan(state, model);
            // Prepare execution plan and warmup
            tornadoVMPlan.executionPlan.withWarmUp();
        }

        // Start timing the whole process
        long startNanos = System.nanoTime();
        long inferenceStartNanos = 0;

        // Validate and adjust maxTokens if necessary
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }

        // Storage for generated tokens
        List<Integer> generatedTokens = new ArrayList<>();

        // Initialize token variables
        int currentToken = state.latestToken;
        int nextToken;
        int promptIndex = 0;
        int pos = startPosition;

        while (pos < maxTokens) {
            // Run the forward pass through the model (CPU or GPU path)
            Object logits;
            if (TornadoVMCompute.TORNADOVM) {
                logits = forwardTornadoVM(model, state, currentToken, pos, tornadoVMPlan);
            } else {
                logits = forwardJava(model, state, currentToken, pos);
            }

            // Handle token processing
            if (promptIndex < promptTokens.size()) {
                // We're still processing the prompt tokens
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    System.err.print(Tokenizer.replaceControlCharacters(
                            model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                // Mark the start of actual generation (after prompt processing)
                if (inferenceStartNanos == 0) {
                    inferenceStartNanos = System.nanoTime();
                }

                // Sample the next token
                nextToken = sampler.sampleToken(logits);

                // Output the token if echo is enabled
                if (echo) {
                    System.err.print(Tokenizer.replaceControlCharacters(
                            model.tokenizer().decode(List.of(nextToken))));
                }

                // Track the generated token
                generatedTokens.add(nextToken);

                // Notify via callback if provided
                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }

                // Check for stop condition
                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }

//            if (startNanos == 0) {
//                startNanos = System.nanoTime();
//            }

            // Update for next iteration
            currentToken = nextToken;
            state.latestToken = currentToken;
            pos++;
        }

        // Calculate and print performance metrics
        long endNanos = System.nanoTime();
        double totalTimeSeconds = (endNanos - startNanos) / 1_000_000_000.0;
        double inferenceTimeSeconds = inferenceStartNanos > 0 ? (endNanos - inferenceStartNanos) / 1_000_000_000.0 : 0;

        int totalTokens = promptIndex + generatedTokens.size();
        int generatedCount = generatedTokens.size();

        // Print C-style performance output
        System.err.printf("\n\nachieved tok/s: %.2f. Tokens: %d, seconds: %.2f\n",
                totalTokens / totalTimeSeconds, totalTokens, totalTimeSeconds);

        // Optional: print more detailed metrics
//        if (inferenceStartNanos > 0) {
//            System.err.printf("Generation speed: %.2f tok/s for %d tokens\n",
//                    generatedCount / inferenceTimeSeconds, generatedCount);
//        }

        // Clean up TornadoVM resources if used
        if (TornadoVMCompute.TORNADOVM) {
            tornadoVMPlan.freeTornadoExecutionPlan();
        }

        return generatedTokens;
    }

    public static List<Integer> generateTokensX(Llama model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        TornadoVMMasterPlan tornadoVMPlan = null;

        if (TornadoVMCompute.TORNADOVM) {
             tornadoVMPlan = new TornadoVMMasterPlan(state, model);
            // Prepare the TornadoVM execution plan -> JIT -> READ-ONLY copy ins
            tornadoVMPlan.executionPlan.withWarmUp();
        }


        long startNanos = System.nanoTime();
        long startGen = 0;
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }

        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;
        for (int position = startPosition; position < maxTokens; ++position) {
            Object logits = runForward(model, state, token, position, tornadoVMPlan);

            // Mark the start of token generation phase (after prompt processing)
            if (promptIndex < promptTokens.size()) {
                // Force-pick token from prompt.
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    // log prompt token (different color?)
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                nextToken = sampler.sampleToken(logits);
                if (echo) {
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
                generatedTokens.add(nextToken);
                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }
                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }

            if (startGen == 0) {
                startGen = System.nanoTime();
            }
            state.latestToken = token = nextToken;
        }
        long elapsedNanos = System.nanoTime() - startGen;
        int totalTokens = promptIndex + generatedTokens.size();

        System.err.printf("\n%n%.2f tokens/s (%d)%n ", (totalTokens-1) / (elapsedNanos / 1_000_000_000.0), totalTokens);

        if (TornadoVMCompute.TORNADOVM) {
            tornadoVMPlan.freeTornadoExecutionPlan();
            // Print TornadoVM performance metrics
        }
        return generatedTokens;
    }



    /**
     * Unified method to run the forward pass through the model
     */
    private static Object runForward(Llama model, State state, int token, int position, TornadoVMMasterPlan tornadoVMPlan) {
        if (TornadoVMCompute.TORNADOVM) {
            return forwardTornadoVM(model, state, token, position, tornadoVMPlan);
        } else {
            return forwardJava(model, state, token, position);
        }
    }

    /**
     * Print performance metrics for the generation process
     */
    private static void printPerformanceMetrics(long startNanos, long inferenceStartNanos, int promptTokenCount, int generatedTokenCount) {
        long endNanos = System.nanoTime();
        long totalNanos = endNanos - startNanos;
        long inferenceNanos = inferenceStartNanos > 0 ? endNanos - inferenceStartNanos : 0;
        long promptNanos = inferenceStartNanos - startNanos;
        int totalTokens = promptTokenCount + generatedTokenCount;

        double totalTokensPerSecond = totalTokens / (totalNanos / 1_000_000_000.0);
        double promptTokensPerSecond = promptTokenCount > 0 ? promptTokenCount / (promptNanos / 1_000_000_000.0) : 0;
        double inferenceTokensPerSecond = generatedTokenCount > 0 ? generatedTokenCount / (inferenceNanos / 1_000_000_000.0) : 0;

        System.err.printf("\n%n%.2f tokens/s (%d) [PrEval %.2f tokens/s (%d), TokGen %.2f tokens/s (%d)]%n", totalTokensPerSecond, totalTokens, promptTokensPerSecond, promptTokenCount,
                inferenceTokensPerSecond, generatedTokenCount);
    }

    public State createNewState() {
        State state = new State(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    public State createNewState(int batchsize) {
        State state = new State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

}

