package com.example.inference.engine.impl;

import com.example.aux.Parallel;
import com.example.core.model.GGMLType;
import com.example.core.model.tensor.FloatTensor;
import com.example.inference.Sampler;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.tokenizer.impl.Tokenizer;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.tensors.Float16;
import uk.ac.manchester.tornado.api.types.tensors.TensorQ8;

import javax.print.DocFlavor;
import javax.xml.transform.Transformer;
import java.lang.foreign.ValueLayout;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public record Llama(Configuration configuration, Tokenizer tokenizer, Weights weights) {
    public static final long WORKGROUP = Long.parseLong(System.getProperty("llama.workgroup", "32"));


    public State createNewState() {
        State state = new State(configuration());
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    static void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        // calculate sum of squares
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }

    static FloatTensor forward(Llama model, State state, int token, int position, TornadoExecutionPlan executionPlan) {
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
            //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
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

        // final rmsnorm
        rmsnorm(state.x, state.x, weights.rms_final_weight, dim, config.rmsNormEps);

        // classifier into logits
//        weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim);
        // Tornado types host code      matmul(weights.wclsTornadoQ8, state.x, state.logits, config.vocabularySize, dim);

        // Update state.x
//        state.wrapXFloat.getSegment().copyFrom(state.x.asMemorySegment());
//        matmul(weights.wclsTornadoQ8, state.wrapXFloat, state.wrapLogits, config.vocabularySize, dim);
//        matmul(weights.wclsByteArray, state.wrapXFloat, state.wrapLogits, config.vocabularySize, dim);
//        state.logits.asMemorySegment().copyFrom(state.wrapLogits.getSegment());
        // This should be replaced with a Tornado call

        // Once fuse or JIT is done we move into the following block

        state.wrapXFloat.getSegment().copyFrom(state.x.asMemorySegment());

        WorkerGrid worker = new WorkerGrid1D(model.configuration.vocabularySize);
        worker.setLocalWork(WORKGROUP,1,1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);
        executionPlan.withGridScheduler(gridScheduler).execute();

        state.logits.asMemorySegment().copyFrom(state.wrapLogits.getSegment());



        return state.logits;
    }

    // state.x.size() -> 2048
    // state.logits.size() -> 2048
    // config.vocabularySize -> 128256
    // dim -> 2048
    public static void matmul(TensorQ8 thisx, FloatArray that, FloatArray out, int dim0, int dim1) {
        IntStream.range(0, dim0).parallel().forEach(i -> {
            float result = 0f;
            int thisOffset = i * dim1;
            for (int j = 0; j < dim1; j++) {
                result += thisx.getFloat(thisOffset + j) * that.get(j);
            }
            out.set(i, result);
        });
    }

    public static void matmulTornadoOptimized(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
            final int BLOCK_SIZE = 32;
            final int BYTES_PER_BLOCK = 34; // 2 + BLOCK_SIZE
            final int TILE_SIZE = (int) WORKGROUP;

            float[] that_cache = context.allocateFloatLocalArray(TILE_SIZE);
            int localId = context.localIdx;
            int globalId = context.globalIdx;

            float[] results = new float[4];

            for (int tile = 0; tile < dim1; tile += TILE_SIZE) {
                // Load collaborative data into local memory
                if (tile + localId < dim1) {
                    that_cache[localId] = that.get(tile + localId);
                }
                context.localBarrier();

                // Process 4 elements per thread
                for (int k = 0; k < 4; k++) {
                    int row = globalId * 4 + k;
                    int thisOffset = row * dim1;

                    for (int j = 0; j < Math.min(TILE_SIZE, dim1 - tile); j++) {
                        int index = thisOffset + tile + j;
                        int blockIndex = index / BLOCK_SIZE;
                        int withinBlockIndex = index % BLOCK_SIZE;
                        int blockOffset = blockIndex * BYTES_PER_BLOCK;

                        // Read and decode scale (float16)
                        int scaleByte1 = thisx.get(blockOffset) & 0xFF;
                        int scaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
                        short scaleFloat16 = (short)((scaleByte2 << 8) | scaleByte1);
                        float scale = decodeFloat16(scaleFloat16);

                        byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);
                        results[k] += (quantized * scale) * that_cache[j];
                    }
                }
                context.localBarrier();
            }

            // Write results
            for (int k = 0; k < 4; k++) {
                int row = globalId * 4 + k;
                if (row < dim1) {
                    out.set(row, results[k]);
                }
            }
        }


//    public static void matmulTornadoOptimized(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
//        final int BLOCK_SIZE = 32;
//        final int BYTES_PER_BLOCK = 2 + BLOCK_SIZE;
//        final int TILE_SIZE = 16; // For shared memory tiling
//
//        int idx = context.globalIdx;
//        float[] thatTile = context.allocateFloatLocalArray(TILE_SIZE);
//        float[] scaleCache = context.allocateFloatLocalArray(TILE_SIZE / BLOCK_SIZE + 1);
//        byte[] quantCache = context.allocateCharLocalArray(TILE_SIZE);
//
//        float result = 0f;
//        int thisOffset = idx * dim1;
//
//        // Process in tiles
//        for (int tile = 0; tile < dim1; tile += TILE_SIZE) {
//            // Collaborative loading of that array into shared memory
//            if (context.localIdx < TILE_SIZE && (tile + context.localIdx) < dim1) {
//                thatTile[context.localIdx] = that.get(tile + context.localIdx);
//            }
//
//            // Pre-fetch scales and quantized values for this tile
//            int localIdx = context.localIdx;
//            if (localIdx < (TILE_SIZE / BLOCK_SIZE + 1)) {
//                int blockIndex = (thisOffset + tile + localIdx * BLOCK_SIZE) / BLOCK_SIZE;
//                int blockOffset = blockIndex * BYTES_PER_BLOCK;
//
//                // Decode scale
//                int scaleByte1 = thisx.get(blockOffset) & 0xFF;
//                int scaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
//                scaleCache[localIdx] = decodeFloat16((short)((scaleByte2 << 8) | scaleByte1));
//
//                // Cache quantized values
//                for (int i = 0; i < BLOCK_SIZE && (i + localIdx * BLOCK_SIZE) < TILE_SIZE; i++) {
//                    quantCache[localIdx * BLOCK_SIZE + i] = thisx.get(blockOffset + 2 + i);
//                }
//            }
//
//            context.localBarrier();
//
//            // Process the tile
//            for (int j = 0; j < TILE_SIZE && (tile + j) < dim1; j++) {
//                int index = thisOffset + tile + j;
//                int blockIndex = index / BLOCK_SIZE;
//                int withinBlockIndex = index % BLOCK_SIZE;
//
//                float scale = scaleCache[blockIndex % (TILE_SIZE / BLOCK_SIZE + 1)];
//                byte quantized = quantCache[blockIndex * BLOCK_SIZE + withinBlockIndex];
//
//                result += (quantized * scale) * thatTile[j];
//            }
//
//            context.localBarrier();
//        }
//
//        out.set(idx, result);
//    }

//    public static void matmulTornadoOptimized(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
//        final int BLOCK_SIZE = 32;
//        final int BYTES_PER_BLOCK = 2 + BLOCK_SIZE;
//        int idx = context.globalIdx;
//
//        // Allocate shared memory for tiles (if the API supports it)
//        float[] thatTile = context.allocateFloatLocalArray(BLOCK_SIZE);
//        float result = 0f;
//        int thisOffset = idx * dim1;
//
//        // Iterate over blocks
//        for (int j = 0; j < dim1; j += BLOCK_SIZE) {
//            // Load a tile of `that` into shared memory
//            int tileEnd = Math.min(j + BLOCK_SIZE, dim1);
//            for (int tj = 0; tj < tileEnd - j; tj++) {
//                thatTile[tj] = that.get(j + tj);
//            }
//            context.localBarrier(); // Ensure all threads have loaded the tile
//
//            // Process each block within the tile
//            for (int k = 0; k < tileEnd - j; k++) {
//                int index = thisOffset + j + k;
//                int blockIndex = index / BLOCK_SIZE;
//                int withinBlockIndex = index % BLOCK_SIZE;
//                int blockOffset = blockIndex * BYTES_PER_BLOCK;
//
//                // Decode the scale (float16) for this block
//                float scale = decodeFloat16(
//                        (short)((thisx.get(blockOffset + 1) & 0xFF) << 8 | (thisx.get(blockOffset) & 0xFF))
//                );
//
//                // Read quantized value and dequantize
//                byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);
//                result += (quantized * scale) * thatTile[k];
//            }
//            context.localBarrier(); // Synchronize before loading the next tile
//        }
//
//        out.set(idx, result);
//    }

    public static void matmulTornado(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
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
            short scaleFloat16 = (short)((scaleByte2 << 8) | scaleByte1);
            float scale = decodeFloat16(scaleFloat16);

            // Read quantized value
            byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);

            // Dequantize and multiply
            result += (quantized * scale) * that.get(j);
        }

        out.set(idx, result);

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
                short scaleFloat16 = (short)((scaleByte2 << 8) | scaleByte1);
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

    private static float decodeFloat16(short value) {
        int sign = (value & 0x8000) >>> 15;
        int exp = (value & 0x7C00) >>> 10;
        int frac = value & 0x03FF;

       // Handle special cases
        if (exp == 0x1F) return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        if (exp == 0) {
            if (frac == 0) return sign == 0 ? 0.0f : -0.0f;
            float result = frac * pow2(-24);
            return sign == 0 ? result : -result;
        }

        float result = 1.0f + (frac / 1024.0f);
        result *= pow2(exp - 15);
        return sign == 0 ? result : -result;
    }



    private static float getFloatFromByteArray(int index, ByteArray data) {
        // Direct read of two bytes at the index position - no multiplication by BYTES_PER_FLOAT16
        int byte1 = data.get(index) & 0xFF;
        int byte2 = data.get(index + 1) & 0xFF;
        short float16Value = (short)((byte2 << 8) | byte1);

        return decodeFloat16(float16Value);
    }

    private static float decodeFloat162(short value) {
        int sign = (value & 0x8000) >>> 15;
        int exp = (value & 0x7C00) >>> 10;
        int frac = value & 0x03FF;

        // Optimized special cases handling
        if (exp == 0x1F) return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        if (exp == 0) {
            if (frac == 0) return sign == 0 ? 0.0f : -0.0f;
            // Subnormal numbers
            float result = frac * pow2(-24);
            return sign == 0 ? result : -result;
        }

        // Normal numbers - optimized calculation
        float result = 1.0f + (frac / 1024.0f);
        result *= pow2(exp - 15);
        return sign == 0 ? result : -result;
    }


        /**
         * Compute 2^n efficiently
         */
        private static float pow2(int n) {
            if (n >= 0) {
                if (n < 31) {
                    return (float)(1 << n);
                }
                return Float.POSITIVE_INFINITY;
            }
            if (n > -150) {
                return 1.0f / (1 << -n);
            }
            return 0.0f;
        }

    /**
     * LLM generation entry point, ingest prompt tokens and generates new tokens.
     *
     * <p>
     * All prompt tokens are ingested first, then inference starts, until a stop token is found.
     * The returned tokens only include generated/inferred tokens.
     *
     * @param model            model to run inference (including weights, configuration, tokenizer ...)
     * @param state            state of the model e.g. key/value caches ... this is mutated by this call
     * @param startPosition    start prompt ingestion + inference at this position in the context e.g. useful if state was kept across calls (chained generation). 0 implies run with no previous context.
     * @param promptTokens     prompt tokens to ingest, all the prompt tokens will be ingested, given there's enough capacity left in the context
     * @param stopTokens       set of tokens that abort generation during inference, stop tokens do not affect prompt ingestion
     * @param maxTokens        maximum number of tokens (can go up to {@link Configuration#contextLength context length}
     *                         if this value is negative or greater than {@link Configuration#contextLength context length}
     * @param sampler          {@link Sampler strategy} used to select tokens
     * @param echo             debugging flag, prints ALL, prompt and inferred tokens, to {@link System#err stderr}
     * @param onTokenGenerated callback, if non-null, it's called every time a token is inferred e.g. it's not called when ingesting prompt tokens
     * @return list of generated/inferred tokens, including the stop token, if any e.g. does not include any token from the prompt
     */
    public static List<Integer> generateTokens(Llama model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;

        // Revert when end-to-end integration is done
        TornadoExecutionPlan tornadoExecutionPlan = createTornadoExecutionPlan(state, model);
//        TornadoExecutionPlan tornadoExecutionPlan = null;

        for (int position = startPosition; position < maxTokens; ++position) {
            forward(model, state, token, position, tornadoExecutionPlan);
            if (promptIndex < promptTokens.size()) {
                // Force-pick token from prompt.
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    // log prompt token (different color?)
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                nextToken = sampler.sampleToken(state.logits);
                if (echo) {
                    // log inferred token
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
            state.latestToken = token = nextToken;
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        int totalTokens = promptIndex + generatedTokens.size();
        System.err.printf("%n%.2f tokens/s (%d)%n", totalTokens / (elapsedNanos / 1_000_000_000.0), totalTokens);

        return generatedTokens;
    }

    /**
     * Creates the appropriate TornadoVM execution plan based on the selected execution mode
     *
     * *            The Transformer model used for sequence generation.
     * @return The TornadoVM execution plan for the specified execution mode
     */
    private static TornadoExecutionPlan createTornadoExecutionPlan(State state, Llama model) {

        TaskGraph taskGraph;

        KernelContext context = new KernelContext();

        taskGraph = new TaskGraph("s0") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapXFloat) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, model.weights.wclsByteArray) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, model.configuration.dim) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, model.configuration.vocabularySize) //
//                .task("t0", Llama:: matmulTornado, context, model.weights.wclsByteArray, state.wrapXFloat, state.wrapLogits, model.configuration.vocabularySize, model.configuration.dim) //
                .task("t0", Llama:: matmulTornadoOptimized, context, model.weights.wclsByteArray, state.wrapXFloat, state.wrapLogits,  model.configuration.dim) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        return new TornadoExecutionPlan(taskGraph.snapshot());
    }
}

