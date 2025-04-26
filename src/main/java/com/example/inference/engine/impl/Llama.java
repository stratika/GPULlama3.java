package com.example.inference.engine.impl;

import com.example.aux.Parallel;
import com.example.core.model.tensor.FloatTensor;
import com.example.inference.Sampler;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tornadovm.TornadoVMCompute;
import com.example.tornadovm.TornadoVMMasterPlan;

import java.lang.foreign.MemorySegment;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public record Llama(Configuration configuration, Tokenizer tokenizer, Weights weights) {

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

    public static FloatTensor forwardJavaDebug2(Llama model, State state, int token, int position, int ll) {
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

        System.out.println("\n==== Java Input State ====");
        System.out.println("First 15 values of x tensor:");
        for (int i = 0; i < 15; i++) {
            System.out.printf("x[%d] = %f%n", i, state.x.getFloat(i));
        }

        // forward all the layers
        for (int l = 0; l < ll; l++) {
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

        System.out.println("\n==== Intermediate State ====");
        System.out.println("First 5 values of x tensor:");
        for (int i = 0; i < 5; i++) {
            System.out.printf("x[%d] = %f%n", i, state.x.getFloat(i));
        }

        System.out.println("\nFirst 5 values of xb tensor:");
        for (int i = 0; i < 5; i++) {
            System.out.printf("xb[%d] = %f%n", i, state.xb.getFloat(i));
        }

        System.out.println("\nFirst 5 values of q tensor:");
        for (int i = 0; i < 5; i++) {
            System.out.printf("q[%d] = %f%n", i, state.q.getFloat(i));
        }

        //        rmsnorm(state.x, state.x, weights.rms_final_weight, dim, config.rmsNormEps);
        //
        //        weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim);

        return state.logits;
    }

    /**
     * Modified version of forwardJavaDebug with comprehensive print statements
     */
    public static FloatTensor forwardJavaDebug(Llama model, State state, int token, int position, int ll) {
        // a few convenience variables
        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);
        System.out.println("\n==== Java State ==== + Position " + position + " Token " + token);

        System.out.println("\n======== JAVA DEBUG START ========");

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        System.out.println("\n==== Initial Token Embedding ====");
        System.out.println("First 15 values of x tensor after token embedding:");
        for (int i = 0; i < 15; i++) {
            System.out.printf("x[%d] = %f%n", i, state.x.getFloat(i));
        }

        // forward all the layers
        for (int l = 0; l < ll; l++) {
            System.out.println("\n=========== Processing LAYER " + l + " ===========");

            // attention rmsnorm
            System.out.println("\n-- Attention RMS Norm --");
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], dim, config.rmsNormEps);
            System.out.println("First 15 values after attention rmsnorm:");
            for (int i = 0; i < 10; i++) {
                System.out.printf("xb[%d] = %f%n", i, state.xb.getFloat(i));
            }
            // qkv matmuls for this position
            System.out.println("\n-- QKV Matmuls --");
            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            System.out.println("First 10 values after Q matmul:");
            for (int i = 0; i < 15; i++) {
                System.out.printf("q[%d] = %f, k[%d] = %f, v[%d] = %f%n", i, state.q.getFloat(i), i, state.k.getFloat(i), i, state.v.getFloat(i));
            }

            // RoPE relative positional encoding
            System.out.println("\n-- RoPE Rotation --");
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

                    if (i < 10) {
                        String vecName = (v == 0) ? "q" : "k";
                        System.out.printf("After RoPE %s[%d]=%f, %s[%d]=%f%n", vecName, i, vec.getFloat(i), vecName, i + 1, vec.getFloat(i + 1));
                    }
                }
            }

            System.out.println("After RoPE - First 10 values of q, k tensors (rotated):");
            for (int i = 0; i < 15; i++) {
                System.out.printf("wrapQ[%d] = %f, wrapK[%d] = %f%n", i, state.q.getFloat(i), i, state.k.getFloat(i));
            }

            // save key,value at this time step to kv cache
            System.out.println("\n-- Copy to KV Cache --");
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            //            System.out.println("First 15 values in key & value cache at position " + position + ":");
            //            for (int i = 0; i < 15; i++) {
            //                System.out.printf(" keyCache[%d] = %f, valueCache[%d] = %f%n", i, state.keyCache[l].getFloat(position * kvDim + i), i, state.valueCache[l].getFloat(position * kvDim + i));
            //            }

            // Print the first 15 values after copying from original tensors
            System.out.println("First 15 values in key & value cache at position " + position + ":");
            for (int i = 0; i < 15; i++) {
                int offset = position * model.configuration.kvDim + i;
                System.out.printf(" keyCache[%d] = %f, valueCache[%d] = %f%n", i, state.keyCache[l].getFloat(offset), i, state.valueCache[l].getFloat(offset));
            }

            // Also print with layer offset for comparison
            int layerOffset = l * model.configuration.contextLength * model.configuration.kvDim;
            int fullOffset = layerOffset + position * model.configuration.kvDim;
            System.out.println("\nFirst 15 values with layer offset (for comparison with wrapped version):");
            for (int i = 0; i < 15; i++) {
                // Note: For tensor arrays, the layer is already selected by [l]
                // so we only need position offset within that layer
                System.out.printf(" keyCache[%d] = %f, valueCache[%d] = %f%n", i, state.keyCache[l].getFloat(position * model.configuration.kvDim + i), i,
                        state.valueCache[l].getFloat(position * model.configuration.kvDim + i));
            }

            // Print wrapped version for comparison
            System.out.println("\nWrapped version values:");
            for (int i = 0; i < 15; i++) {
                System.out.printf(" wrapKeyCache[%d] = %f, wrapValueCache[%d] = %f%n", i, state.wrapKeyCache.get(fullOffset + i), i, state.wrapValueCache.get(fullOffset + i));
            }

            int curLayer = l;

            // Track attention scores and weights for debugging
            float[][] attScores = new float[config.numberOfHeads][position + 1];
            float[][] attWeights = new float[config.numberOfHeads][position + 1];

            // multihead attention
            System.out.println("\n-- Multi-Head Attention --");
            Parallel.parallelFor(0, config.numberOfHeads, h -> {
                int qOffset = h * headSize;
                int attOffset = h * config.contextLength;

                // Debug for first few heads only to avoid excessive output
                //                boolean debugThisHead = h < 2;  // Only debug first 2 heads
                boolean debugThisHead = false;  // Only debug first 2 heads

                if (debugThisHead) {
                    System.out.println("Head " + h + " processing:");
                    System.out.println("Query offset: " + qOffset + ", Attention offset: " + attOffset);
                }

                for (int t = 0; t <= position; t++) {
                    int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    state.att.setFloat(attOffset + t, score);

                    if (debugThisHead && t <= 5) {  // Limit to first few positions
                        System.out.printf("  Position %d: Raw attention score = %f%n", t, score);
                        attScores[h][t] = score;  // Store for later printing
                    }
                }

                // softmax
                state.att.softmaxInPlace(attOffset, position + 1);

                if (debugThisHead) {
                    System.out.println("  After softmax:");
                    for (int t = 0; t <= Math.min(5, position); t++) {
                        float weight = state.att.getFloat(attOffset + t);
                        System.out.printf("  Position %d: Attention weight = %f%n", t, weight);
                        attWeights[h][t] = weight;  // Store for later printing
                    }
                }

                // weighted sum of values
                int xbOffset = h * headSize;
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    int vOffset = t * kvDim + (h / kvMul) * headSize;
                    float a = state.att.getFloat(attOffset + t);
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }

                if (debugThisHead) {
                    System.out.println("  After weighted sum, first 5 values in xb:");
                    for (int i = 0; i < 5; i++) {
                        System.out.printf("  xb[%d] = %f%n", xbOffset + i, state.xb.getFloat(xbOffset + i));

                    }
                }
            });

            for (int i = 0; i < 15; i++) {
                System.out.printf("  xb[%d] = %f%n", i, state.xb.getFloat(i));
            }

            // final matmul to get output of attention
            System.out.println("\n-- Attention Output Projection --");
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            System.out.println("First 10 values after wo matmul:");
            //            for (int i = 0; i < 15; i++) {
            //                System.out.printf("xb2[%d] = %f%n", i, state.xb2.getFloat(i));
            //            }
            state.x.addInPlace(state.xb2);
            // residual connection back into x
            System.out.println("\n-- First Residual Connection --");

            System.out.println("\nFirst 15 values of x after residual:");
            for (int i = 0; i < 15; i++) {
                System.out.printf("x[%d] = %f, xb2[%d] = %f%n", i, state.x.getFloat(i), i, state.xb2.getFloat(i));
            }

            //
            // ffn rmsnorm
            System.out.println("\n-- FFN RMS Norm --");

            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], dim, config.rmsNormEps);

            System.out.println("First 10 values after FFN rmsnorm:");
            for (int i = 0; i < 15; i++) {
                System.out.printf("xb[%d] = %f%n", i, state.xb.getFloat(i));
            }

            // FFN forward pass
            System.out.println("\n-- FFN Forward Pass --");
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim, dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim, dim);

            System.out.println("First 10 values after w1 matmul (hb):");
            for (int i = 0; i < 15; i++) {
                System.out.printf("hb[%d] = %f, hb2[%d] = %f%n", i, state.hb.getFloat(i), i, state.hb2.getFloat(i));
            }

            // SwiGLU non-linearity
            System.out.println("\n-- SwiGLU Activation --");

            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            // elementwise multiply with w3(x)
            System.out.println("\n-- Elementwise Multiply  --");
            state.hb.multiplyInPlace(state.hb2);

            System.out.println("First 14 values after silu_elementwise_mul:");
            for (int i = 0; i < 15; i++) {
                System.out.printf("hb[%d] = %f%n", i, state.hb.getFloat(i));
            }

            // final matmul to get FFN output
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim);

            // residual connection
            //            System.out.println("\n-- Second Residual Connection --");
            System.out.println("\n-- FFN Output Projection --");

            state.x.addInPlace(state.xb);

            System.out.println("\nFirst 10 values of x after residual:");
            for (int i = 0; i < 15; i++) {
                System.out.printf("x[%d] = %f, xb[%d] = %f%n", i, state.x.getFloat(i), i, state.xb.getFloat(i));
            }

            System.out.println("\n========= END PROCESSING LAYER " + l + " =========");
        }
        System.out.println("\n======== Logits ========");
        rmsnorm(state.x, state.x, weights.rms_final_weight, dim, config.rmsNormEps);

        weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim);

        for (int i = 0; i < 10; i++) {
            System.out.printf("x[%d] = %f%n", i, state.x.getFloat(i));
        }

        int totalSize = state.logits.size();
        int step = Math.max(1, totalSize / 20);  // 1/20 = 5%

        for (int i = 0; i < totalSize; i += step) {
            System.out.printf("logits[%d] = %f%n", i, state.logits.getFloat(i));
        }

        System.out.println("\n======== JAVA DEBUG END ========");
        return state.logits;
    }

    public static FloatTensor forwardTornadoVM( //
            Llama model,  //
            State state,  //
            int token,    //
            int position,   //
            TornadoVMMasterPlan tornadoVMMasterPlan) { //

        model.weights.token_embedding_table.copyTo(token * model.configuration.dim, state.x, 0, model.configuration.dim);

        MemorySegment.copy(state.x.asMemorySegment(), 0, state.wrapX.getSegment(), 0, model.configuration.dim * Float.BYTES);

        tornadoVMMasterPlan.executionPlan.withGraph(0).withGridScheduler(tornadoVMMasterPlan.scheduler).execute();

        for (int layer = 0; layer < model.configuration.numberOfLayers; layer++) {
            int loff = layer * model.configuration.contextLength * model.configuration.kvDim;

            int layerOffsetForCaches = loff + position * model.configuration.kvDim;

            state.positionAndLayer.set(0, position);
            state.positionAndLayer.set(1, layer);
            state.positionAndLayer.set(2, layerOffsetForCaches);
            state.positionAndLayer.set(3, loff);

            tornadoVMMasterPlan.executionPlan.withGraph(1).withGridScheduler(tornadoVMMasterPlan.scheduler).execute();

        }

        tornadoVMMasterPlan.executionPlan.withGraph(2).withGridScheduler(tornadoVMMasterPlan.scheduler).execute();

        state.logits.asMemorySegment().copyFrom(state.wrapLogits.getSegment());

        return state.logits;
    }

    public static List<Integer> generateTokens(Llama model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {

        TornadoVMMasterPlan tornadoVMPlan = new TornadoVMMasterPlan(state, model);

        // Prepare the TornadoVM execution plan -> JIT -> READ-ONLY copy ins
        tornadoVMPlan.executionPlan.withWarmUp();

        long startNanos = System.nanoTime();
        long startGen = 0;
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;
        int counter = 0;
        for (int position = startPosition; position < maxTokens; ++position) {
            if (TornadoVMCompute.TORNADOVM) {
                forwardTornadoVM(model, state, token, position, tornadoVMPlan);
            } else {
                forwardJava(model, state, token, position);
            }
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

        // Free the TornadoVM execution plan
        //        tornadoVMPlan.freeTornadoExecutionPlan();

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

