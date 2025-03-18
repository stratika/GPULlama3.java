package com.example.inference.engine.impl;

import com.example.aux.Parallel;
import com.example.aux.Tuple2;
import com.example.core.model.tensor.FloatTensor;
import com.example.inference.Sampler;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tornadovm.TornadoVMCompute;
import com.example.tornadovm.TornadoVMLayerPlanner;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.lang.foreign.MemorySegment;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public record Llama(Configuration configuration, Tokenizer tokenizer, Weights weights) {

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

    static FloatTensor forward(Llama model, State state, int token, int position, Tuple2<List<ImmutableTaskGraph>, GridScheduler> tornadoVMListOfPlan) {
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
            //            weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim);

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

    static FloatTensor forwardTornadoVMExplicitCopy(Llama model, State state, int token, int position, Tuple2<List<ImmutableTaskGraph>, GridScheduler> tornadoVMListOfPlan) {

        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int seqLen = config.headSize;
        System.out.println("=== Starting Explicit Copy TornadoVM Forward Pass ===");
        FloatArray attScores = new FloatArray(seqLen);
        FloatArray maxValues = new FloatArray(1);
        FloatArray expValues = new FloatArray(seqLen);
        FloatArray sumValues = new FloatArray(1);

        // Copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);
        System.out.println("Token embedding copied to state.x");

        // Explicitly copy to wrapX
        MemorySegment.copy(state.x.asMemorySegment(), 0, state.wrapX.getSegmentWithHeader(), 0, dim * 4);
        System.out.println("state.x copied to state.wrapX");

        // Set position
        state.positionAndLayer.set(0, position);
        System.out.println("Position set to: " + position);

        List<GridScheduler> schedulers = tornadoVMListOfPlan.getSecond();
        List<ImmutableTaskGraph> taskGraphs = tornadoVMListOfPlan.getFirst();

        // Process just one layer for testing
        int l = 0;
        state.positionAndLayer.set(1, l);
        System.out.println("Layer set to: " + l);

        // Step 1: Execute lookUpBufferX
        System.out.println("\nExecuting Graph 0: lookUpBufferX");
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(0))) {
            executionPlan.withGridScheduler(schedulers.get(0)).execute();
            System.out.println("✅ Graph 0 executed successfully");
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 0: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // Step 2: Execute RMSNorm
        System.out.println("\nExecuting Graph 1: RMSNorm");
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(1))) {
            executionPlan.withGridScheduler(schedulers.get(1)).execute();
            System.out.println("✅ Graph 1 executed successfully");

            // Debug: Copy wrapXb back to verify results
            System.out.println("Verifying state.wrapXb after RMSNorm");
            FloatArray tempXb = new FloatArray(dim);
            MemorySegment.copy(state.wrapXb.getSegment(), 0, tempXb.getSegmentWithHeader(), 0, dim * 4);
            float sum = 0;
            for (int i = 0; i < Math.min(5, dim); i++) {
                sum += tempXb.get(i);
                System.out.println("wrapXb[" + i + "] = " + tempXb.get(i));
            }
            System.out.println("Average of first 5 values: " + (sum / Math.min(5, dim)));
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 1: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // Step 3: Execute QKV Matmuls
        System.out.println("\nExecuting Graph 2: QKV Matmuls");
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(2))) {
            executionPlan.withGridScheduler(schedulers.get(2)).execute();
            System.out.println("✅ Graph 2 executed successfully");

            // Debug: Copy Q, K, V back to verify results
            System.out.println("Verifying state.wrapQ after QKV Matmuls");
            FloatArray tempQ = new FloatArray(dim);
            MemorySegment.copy(state.wrapQ.getSegment(), 0, tempQ.getSegmentWithHeader(), 0, dim * 4);
            for (int i = 0; i < Math.min(5, dim); i++) {
                System.out.println("wrapQ[" + i + "] = " + tempQ.get(i));
            }
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 2: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // Step 4: Execute RoPE
        System.out.println("\nExecuting Graph 3: RoPE");
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(3))) {
            executionPlan.withGridScheduler(schedulers.get(3)).execute();
            System.out.println("✅ Graph 3 executed successfully");

            // Debug: Copy Q, K back to verify results after rotation
            System.out.println("Verifying state.wrapQ after RoPE");
            FloatArray tempQ = new FloatArray(dim);
            MemorySegment.copy(state.wrapQ.getSegment(), 0, tempQ.getSegmentWithHeader(), 0, dim * 4);
            for (int i = 0; i < Math.min(10, dim); i += 2) {
                System.out.println("wrapQ[" + i + "," + (i + 1) + "] = " + tempQ.get(i) + ", " + tempQ.get(i + 1));
            }
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 3: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // Step 5: Copy Q, K to KV cache manually
        long offset = l * config.contextLength * kvDim + position * kvDim;
        System.out.println("\nManually copying Q/K to KV cache at offset: " + offset);

        // Create a temporary copy for the KV cache
        FloatArray tempK = new FloatArray(dim);
        MemorySegment.copy(state.wrapK.getSegment(), 0, tempK.getSegmentWithHeader(), 0, dim * 4);

        // Copy to KV cache
        for (int i = 0; i < kvDim; i++) {
            state.keyCache[l].setFloat(position * kvDim + i, tempK.get(i));
        }
        System.out.println("K manually copied to KV cache");

        // Create a temporary copy for the V values
        FloatArray tempV = new FloatArray(dim);
        MemorySegment.copy(state.wrapV.getSegment(), 0, tempV.getSegmentWithHeader(), 0, dim * 4);

        // Copy to KV cache
        for (int i = 0; i < kvDim; i++) {
            state.valueCache[l].setFloat(position * kvDim + i, tempV.get(i));
        }
        System.out.println("V manually copied to KV cache");

        // Step 6: Create a standalone Attention graph that doesn't rely on memory mapping
        System.out.println("\nExecuting standalone Attention graph");
        try {
            // Create the standalone task graph
            TaskGraph standaloneAttention = new TaskGraph("standalone_attention").transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionAndLayer, state.wrapQ, state.wrapKeyCache,
                            state.wrapValueCache)
                    .task("scores", TornadoVMCompute::calculateAttentionScores, new KernelContext(), state.positionAndLayer, config.contextLength, state.wrapQ, state.wrapKeyCache, state.wrapAtt,
                            kvDim, config.numberOfHeads / config.numberOfKeyValueHeads, config.headSize, 0)
                    .task("max", TornadoVMCompute::findMaxAttentionScores, new KernelContext(), state.positionAndLayer, config.contextLength, state.wrapAtt, maxValues, 64)
                    .task("expsum", TornadoVMCompute::calculateExpAndSum, new KernelContext(), state.positionAndLayer, config.contextLength, state.wrapAtt, maxValues, expValues, sumValues, 64)
                    .task("normalize", TornadoVMCompute::normalizeSoftmax, new KernelContext(), state.positionAndLayer, config.contextLength, expValues, sumValues, state.wrapAtt)
                    .task("weighted-sum", TornadoVMCompute::computeWeightedSum, new KernelContext(), state.positionAndLayer, config.contextLength, state.wrapAtt, state.wrapValueCache, state.wrapXb,
                            kvDim, config.numberOfHeads / config.numberOfKeyValueHeads, config.headSize, 0).transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapXb);

            ImmutableTaskGraph immutableStandaloneGraph = standaloneAttention.snapshot();

            // Create a scheduler for the standalone graph
            GridScheduler standaloneScheduler = new GridScheduler();
            WorkerGrid headsWorker = new WorkerGrid1D(config.numberOfHeads * config.headSize);
            headsWorker.setGlobalWork(config.numberOfHeads * config.headSize, 1, 1);
            headsWorker.setLocalWork(64, 1, 1);
            standaloneScheduler.addWorkerGrid("standalone_attention.scores", headsWorker);
            standaloneScheduler.addWorkerGrid("standalone_attention.max", headsWorker);
            standaloneScheduler.addWorkerGrid("standalone_attention.expsum", headsWorker);
            standaloneScheduler.addWorkerGrid("standalone_attention.normalize", headsWorker);
            standaloneScheduler.addWorkerGrid("standalone_attention.weighted-sum", headsWorker);

            // Execute the standalone graph
            try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableStandaloneGraph)) {
                executionPlan.withGridScheduler(standaloneScheduler).execute();
                System.out.println("✅ Standalone attention graph executed successfully");

                // Debug: Copy Xb back to verify results
                System.out.println("Verifying state.wrapXb after Attention");
                FloatArray tempXb = new FloatArray(dim);
                MemorySegment.copy(state.wrapXb.getSegment(), 0, tempXb.getSegmentWithHeader(), 0, dim * 4);
                for (int i = 0; i < Math.min(5, dim); i++) {
                    System.out.println("wrapXb[" + i + "] = " + tempXb.get(i));
                }
            }
        } catch (Exception e) {
            System.out.println("❌ Error in standalone attention: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // Step 7: Execute FFN
        System.out.println("\nExecuting Graph 5: FFN");
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(5))) {
            executionPlan.withGridScheduler(schedulers.get(5)).execute();
            System.out.println("✅ Graph 5 executed successfully");

            // Debug: Copy X back to verify results
            System.out.println("Verifying state.wrapX after FFN");
            FloatArray tempX = new FloatArray(dim);
            MemorySegment.copy(state.wrapX.getSegment(), 0, tempX.getSegmentWithHeader(), 0, dim * 4);
            for (int i = 0; i < Math.min(5, dim); i++) {
                System.out.println("wrapX[" + i + "] = " + tempX.get(i));
            }
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 5: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // Step 8: Execute Final RMSNorm
        System.out.println("\nExecuting Graph 6: Final RMSNorm");
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(6))) {
            executionPlan.withGridScheduler(schedulers.get(6)).execute();
            System.out.println("✅ Graph 6 executed successfully");

            // Debug: Copy X back to verify results
            System.out.println("Verifying state.wrapX after Final RMSNorm");
            FloatArray tempX = new FloatArray(dim);
            MemorySegment.copy(state.wrapX.getSegment(), 0, tempX.getSegmentWithHeader(), 0, dim * 4);
            for (int i = 0; i < Math.min(5, dim); i++) {
                System.out.println("wrapX[" + i + "] = " + tempX.get(i));
            }
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 6: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // Step 9: Execute Logits Projection
        System.out.println("\nExecuting Graph 7: Logits Projection");
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(7))) {
            executionPlan.withGridScheduler(schedulers.get(7)).execute();
            System.out.println("✅ Graph 7 executed successfully");

            // Copy results from TornadoVM buffers to state.logits
            state.logits.asMemorySegment().copyFrom(state.wrapLogits.getSegment());

            // Debug: Examine top logits
            System.out.println("Top 5 logits:");
            float maxLogit = Float.NEGATIVE_INFINITY;
            int maxIdx = -1;
            for (int i = 0; i < config.vocabularySize; i++) {
                if (state.logits.getFloat(i) > maxLogit) {
                    maxLogit = state.logits.getFloat(i);
                    maxIdx = i;
                }
            }
            System.out.println("Max logit: " + maxLogit + " at index " + maxIdx);

            System.out.println("\n=== Explicit Copy TornadoVM Forward Pass Completed Successfully ===");
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 7: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        return state.logits;
    }

    static FloatTensor forwardTornadoVMIncremental(Llama model, State state, int token, int position, Tuple2<List<ImmutableTaskGraph>, List<GridScheduler>> tornadoVMListOfPlan) {

        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;

        // Copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);
        MemorySegment.copy(state.x.asMemorySegment(), 0, state.wrapX.getSegmentWithHeader(), 0, dim * 4);

        state.positionAndLayer.set(0, position);

        List<GridScheduler> schedulers = tornadoVMListOfPlan.getSecond();
        List<ImmutableTaskGraph> taskGraphs = tornadoVMListOfPlan.getFirst();

        System.out.println("=== Starting Incremental TornadoVM Forward Pass ===");

        // Test Graph 0: lookUpBufferX
        System.out.println("\nTesting Graph 0: lookUpBufferX");
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(0))) {
            executionPlan.withGridScheduler(schedulers.get(0)).execute();
            System.out.println("✅ Graph 0 executed successfully");
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 0: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // Test Graph 1: RMSNorm
        System.out.println("\nTesting Graph 1: RMSNorm");
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(1))) {
            executionPlan.withGridScheduler(schedulers.get(1)).execute();
            System.out.println("✅ Graph 1 executed successfully");
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 1: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // Process just one layer for testing
        int l = 0;
        state.positionAndLayer.set(1, l);

        // Test Graph 2: QKV Matmuls
        System.out.println("\nTesting Graph 2: QKV Matmuls for layer " + l);
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(2))) {
            executionPlan.withGridScheduler(schedulers.get(2)).execute();
            System.out.println("✅ Graph 2 executed successfully");
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 2: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // Test Graph 3: RoPE
        System.out.println("\nTesting Graph 3: RoPE for layer " + l);
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(3))) {
            executionPlan.withGridScheduler(schedulers.get(3)).execute();
            System.out.println("✅ Graph 3 executed successfully");
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 3: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // Memory mapping offset calculation
        long offset = l * config.contextLength * kvDim + position * kvDim;
        System.out.println("\nMapping memory regions at offset: " + offset);
        System.out.println("Key cache size: " + state.wrapKeyCache.getSize());
        System.out.println("K vector size: " + state.wrapK.getSize());

        // Test Graph 4: Attention - create a temporary test graph to avoid memory mapping
        System.out.println("\nTesting Graph 4: Attention for layer " + l);

        try {
            // Create a simplified test task graph that doesn't rely on memory mapping
            TaskGraph testAttentionGraph = new TaskGraph("test_attention").transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionAndLayer, state.wrapQ, state.wrapKeyCache)
                    .task("scores", TornadoVMCompute::calculateAttentionScores, new KernelContext(), state.positionAndLayer, config.contextLength, state.wrapQ, state.wrapKeyCache, state.wrapAtt,
                            kvDim, config.numberOfHeads / config.numberOfKeyValueHeads, config.headSize, 0).transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapAtt);

            ImmutableTaskGraph immutableTestGraph = testAttentionGraph.snapshot();

            // Create a scheduler for the test graph
            GridScheduler testScheduler = new GridScheduler();
            WorkerGrid headsWorker = new WorkerGrid1D(config.numberOfHeads * config.headSize);
            headsWorker.setGlobalWork(config.numberOfHeads * config.headSize, 1, 1);
            headsWorker.setLocalWork(64, 1, 1);
            testScheduler.addWorkerGrid("test_attention.scores", headsWorker);

            // Execute the test graph
            try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTestGraph)) {
                executionPlan.withGridScheduler(testScheduler).execute();
                System.out.println("✅ Test attention scores calculated successfully");
            }

            System.out.println("✅ Graph 4 alternative test successful");
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 4 alternative: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // If this works, then we can try a multi-graph execution for the actual attention
        try {
            System.out.println("\nTesting multi-graph execution with RoPE and Attention");
            try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(3), taskGraphs.get(4))) {
                // Execute the RoPE graph first
                executionPlan.withGraph(0).withGridScheduler(schedulers.get(3)).execute();

                // Map memory regions between the two graphs
                executionPlan.mapOnDeviceMemoryRegion(state.wrapKeyCache, state.wrapK, offset, 0, 1);
                executionPlan.mapOnDeviceMemoryRegion(state.wrapValueCache, state.wrapV, offset, 0, 1);

                // Execute the Attention graph
                executionPlan.withGraph(1).withGridScheduler(schedulers.get(4)).execute();

                System.out.println("✅ Multi-graph with memory mapping executed successfully");
            }
        } catch (Exception e) {
            System.out.println("❌ Error in multi-graph execution: " + e.getMessage());
            e.printStackTrace();
            // Continue testing even if this fails
        }

        // Test Graph 5: FFN
        System.out.println("\nTesting Graph 5: FFN for layer " + l);
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(5))) {
            executionPlan.withGridScheduler(schedulers.get(5)).execute();
            System.out.println("✅ Graph 5 executed successfully");
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 5: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // Test Graph 6: Final RMSNorm
        System.out.println("\nTesting Graph 6: Final RMSNorm");
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(6))) {
            executionPlan.withGridScheduler(schedulers.get(6)).execute();
            System.out.println("✅ Graph 6 executed successfully");
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 6: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        // Test Graph 7: Logits Projection
        System.out.println("\nTesting Graph 7: Logits Projection");
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraphs.get(7))) {
            executionPlan.withGridScheduler(schedulers.get(7)).execute();
            System.out.println("✅ Graph 7 executed successfully");

            // Copy results from TornadoVM buffers to state.logits
            state.logits.asMemorySegment().copyFrom(state.wrapLogits.getSegment());
            System.out.println("\n=== Incremental TornadoVM Forward Pass Completed Successfully ===");
        } catch (Exception e) {
            System.out.println("❌ Error in Graph 7: " + e.getMessage());
            e.printStackTrace();
            return state.logits;
        }

        return state.logits;
    }

    static FloatTensor forwardTornadoVM(Llama model, State state, int token, int position,  //
            Tuple2<List<ImmutableTaskGraph>, List<GridScheduler>> tornadoVMListOfPlan) { //

        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;

        // copy the token embedding into x

        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);
        MemorySegment.copy(state.x.asMemorySegment(), 0, state.wrapX.getSegmentWithHeader(), 0, dim * 4);

        state.positionAndLayer.set(0, position);

        List<GridScheduler> schedulers = tornadoVMListOfPlan.getSecond();
        List<ImmutableTaskGraph> taskGraphs = tornadoVMListOfPlan.getFirst();

        // @formatter:off
        try (
            TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(
                    taskGraphs.get(0),
                    taskGraphs.get(1),
                    taskGraphs.get(2),
                    taskGraphs.get(3),
                    taskGraphs.get(4),
                    taskGraphs.get(5),
                    taskGraphs.get(6),
                    taskGraphs.get(7))
        ) {
        // @formatter:on
            // Process each layer
            executionPlan.withGraph(0).withGridScheduler(schedulers.get(0)).execute();
            for (int l = 0; l < config.numberOfLayers; l++) {

                state.positionAndLayer.set(1, l); // Update before execute (it an every copy in)

                // Step 1: RMSNorm for attention
                executionPlan.withGraph(1).withGridScheduler(schedulers.get(1)).execute();

                // Step 2: QKV Matmuls
                executionPlan.withGraph(2).withGridScheduler(schedulers.get(2)).execute();

                // Step 3: RoPE rotation
                executionPlan.withGraph(3).withGridScheduler(schedulers.get(3)).execute();

                // new shift by l
                //    // Calculate the offset based on layer, max sequence length, and position
                long offset = l * config.contextLength * kvDim + position * kvDim;
                System.out.println("Mapping memory regions at offset: " + offset);
                System.out.println("Key cache size: " + state.wrapKeyCache.getSize());
                System.out.println("K vector size: " + state.wrapK.getSize());

                executionPlan.mapOnDeviceMemoryRegion(state.wrapKeyCache, state.wrapK, offset, 3, 4);
                executionPlan.mapOnDeviceMemoryRegion(state.wrapValueCache, state.wrapV, offset, 3, 4);

                // Step 4: Multi-head Attention (scores, softmax, weighted sum)
                executionPlan.withGraph(4).withGridScheduler(schedulers.get(4)).execute();

                // Step 5: Feed-forward neural network
                executionPlan.withGraph(5).withGridScheduler(schedulers.get(5)).execute();
            }

            // Final RMSNorm
            executionPlan.withGraph(6).withGridScheduler(schedulers.get(6)).execute();

            // Final projection to logits
            executionPlan.withGraph(7).withGridScheduler(schedulers.get(7)).execute();

            // Copy results from TornadoVM buffers to state.logits
            state.logits.asMemorySegment().copyFrom(state.wrapLogits.getSegment());
        } catch (TornadoExecutionPlanException e) {
            throw new RuntimeException(e);
        }

        // This copy-out after every execution !!!
        state.logits.asMemorySegment().copyFrom(state.wrapLogits.getSegment());
        // this can be simplified as well
        return state.logits;
    }

    /**
     * LLM generation entry point, ingest prompt tokens and generates new tokens.
     *
     * <p>
     * All prompt tokens are ingested first, then inference starts, until a stop token is found. The returned tokens only include generated/inferred tokens.
     *
     * @param model
     *         model to run inference (including weights, configuration, tokenizer ...)
     * @param state
     *         state of the model e.g. key/value caches ... this is mutated by this call
     * @param startPosition
     *         start prompt ingestion + inference at this position in the context e.g. useful if state was kept across calls (chained generation). 0 implies run with no previous context.
     * @param promptTokens
     *         prompt tokens to ingest, all the prompt tokens will be ingested, given there's enough capacity left in the context
     * @param stopTokens
     *         set of tokens that abort generation during inference, stop tokens do not affect prompt ingestion
     * @param maxTokens
     *         maximum number of tokens (can go up to {@link Configuration#contextLength context length} if this value is negative or greater than {@link Configuration#contextLength context length}
     * @param sampler
     *         {@link Sampler strategy} used to select tokens
     * @param echo
     *         debugging flag, prints ALL, prompt and inferred tokens, to {@link System#err stderr}
     * @param onTokenGenerated
     *         callback, if non-null, it's called every time a token is inferred e.g. it's not called when ingesting prompt tokens
     * @return list of generated/inferred tokens, including the stop token, if any e.g. does not include any token from the prompt
     */
    public static List<Integer> generateTokens(Llama model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        TornadoVMLayerPlanner tornadoVMLayerPlanner = new TornadoVMLayerPlanner(state, model);
        Tuple2<List<ImmutableTaskGraph>, GridScheduler> tornadoVMPlan = tornadoVMLayerPlanner.setupTornadoForwardPlan();

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
            //            forward(model, state, token, position, tornadoVMPlan);
            forwardTornadoVMExplicitCopy(model, state, token, position, tornadoVMPlan);
            startGen = System.nanoTime();
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
        long promptNanos = startGen - startNanos;
        long genNanos = elapsedNanos - startGen + startNanos;
        int totalTokens = promptIndex + generatedTokens.size();

        System.err.printf("\n%n%.2f tokens/s (%d) [PrEval %.2f tokens/s (%d), TokGen %.2f tokens/s (%d)]%n", totalTokens / (elapsedNanos / 1_000_000_000.0), totalTokens,
                promptTokens.size() / (promptNanos / 1_000_000_000.0), promptTokens.size(), generatedTokens.size() / (genNanos / 1_000_000_000.0), generatedTokens.size());
        return generatedTokens;
    }

    public State createNewState() {
        State state = new State(configuration());
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

}

