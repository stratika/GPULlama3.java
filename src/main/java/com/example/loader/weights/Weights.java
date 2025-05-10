package com.example.loader.weights;

import com.example.LlamaApp;
import com.example.core.model.tensor.FloatTensor;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.collections.VectorFloat4;
import uk.ac.manchester.tornado.api.types.vectors.Float4;

import java.nio.FloatBuffer;

public final class Weights {
    // token embedding table
    public final FloatTensor token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    public final FloatBuffer[] rms_att_weight; // (layer, dim) rmsnorm weights
    // weights for matmuls
    public final FloatTensor[] wq; // (layer, n_heads * head_size)
    public final FloatTensor[] wk; // (layer, n_kv_heads, head_size)
    public final FloatTensor[] wv; // (layer, n_kv_heads * head_size)
    public final FloatTensor[] wo; // (layer, n_heads * head_size, dim)
    public final FloatBuffer[] rms_ffn_weight; // (layer, dim)

    // weights for ffn
    public final FloatTensor[] w1; // (layer, hidden_dim, dim)
    public final FloatTensor[] w2; // (layer, dim, hidden_dim)
    public final FloatTensor[] w3; // (layer, hidden_dim, dim)

    // Flatten Structure
//    public final FloatArray rms_att_weightFlat; // (layer, dim) rmsnorm weights
//    public final FloatArray wqFlat; // (layer, n_heads * head_size)
//    public final FloatArray wkFlat; // (layer, n_kv_heads, head_size)
//    public final FloatArray wvFlat; // (layer, n_kv_heads * head_size)
//    public final FloatArray woFlat; // (layer, n_heads * head_size, dim)
//    public final FloatArray rms_ffn_weightFlat; // (layer, dim)
//
//    public final FloatArray w1Flat; // (layer, hidden_dim, dim)
//    public final FloatArray w2Flat; // (layer, dim, hidden_dim)
//    public final FloatArray w3Flat; // (layer, hidden_dim, dim)
//
//    // Layered Data structures
    public final FloatArray[] rms_att_weightLayered; // (layer, dim) rmsnorm weights
    public final FloatArray[] wqLayered; // (layer, n_heads * head_size)
    public final FloatArray[] wkLayered; // (layer, n_kv_heads, head_size)
    public final FloatArray[] wvLayered; // (layer, n_kv_heads * head_size)
    public final FloatArray[] woLayered; // (layer, n_heads * head_size, dim)
    public final FloatArray[] rms_ffn_weightLayered; // (layer, dim)

    public final FloatArray[] w1Layered; // (layer, hidden_dim, dim)
    public final FloatArray[] w2Layered; // (layer, dim, hidden_dim)
    public final FloatArray[] w3Layered; // (layer, hidden_dim, dim)

    //
    public final FloatTensor wcls; // (vocab_size, dim)
    public final ByteArray wclsByteArray;
    public final FloatArray rms_final_weight_as_floatArray;

    public final FloatArray tokenEmbeddingTable; // (vocab_size, dim)
    //

    public final FloatArray freq_cis_realFlat; // (seq_len, head_size/2)
    public final FloatArray freq_cis_imagFlat; // (seq_len, head_size/2)


    // public final rmsnorm
    public final FloatBuffer rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    public final FloatBuffer freq_cis_real; // (seq_len, head_size/2)
    public final FloatBuffer freq_cis_imag; // (seq_len, head_size/2)
    // (optional) classifier weights for the logits, on the last layer

    public Weights(FloatTensor token_embedding_table, FloatBuffer[] rms_att_weight, FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo, FloatBuffer[] rms_ffn_weight,
            FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3, FloatBuffer rms_final_weight, FloatBuffer freq_cis_real, FloatBuffer freq_cis_imag, FloatTensor wcls) {
        this.token_embedding_table = token_embedding_table;
        this.rms_att_weight = rms_att_weight;
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;
        this.rms_ffn_weight = rms_ffn_weight;
        this.w1 = w1;
        this.w2 = w2;
        this.w3 = w3;
        this.rms_final_weight = rms_final_weight;
        this.freq_cis_real = freq_cis_real;
        this.freq_cis_imag = freq_cis_imag;
        this.wcls = wcls;
        this.tokenEmbeddingTable = loadToFloatArray(token_embedding_table); // (vocab_size, dim)


        if (LlamaApp.TORNADOVM) {
            this.freq_cis_imagFlat = loadToSingleFloatArray(freq_cis_imag);
            this.freq_cis_realFlat = loadToSingleFloatArray(freq_cis_real);

            // Store read-only weight as a ByteArray in TornadoVM
            this.wclsByteArray = ByteArray.fromSegment(wcls.asMemorySegment());
            this.rms_final_weight_as_floatArray = FloatArray.fromFloatBuffer(rms_final_weight);

            this.rms_att_weightLayered = loadToFloatArray(rms_att_weight);

            this.wqLayered = loadToFloatArray(wq);
            this.wkLayered = loadToFloatArray(wk);
            this.wvLayered = loadToFloatArray(wv);
            this.woLayered = loadToFloatArray(wo);
            this.rms_ffn_weightLayered = loadToFloatArray(rms_ffn_weight);
            this.w1Layered = loadToFloatArray(w1);
            this.w2Layered = loadToFloatArray(w2);
            this.w3Layered = loadToFloatArray(w3);

        } else {
            this.freq_cis_imagFlat = null;
            this.freq_cis_realFlat = null;
            this.wclsByteArray = null;
            this.rms_final_weight_as_floatArray = null;
            this.rms_att_weightLayered = null;
            this.wqLayered = null;
            this.wkLayered = null;
            this.wvLayered = null;
            this.woLayered = null;
            this.rms_ffn_weightLayered = null;
            this.w1Layered = null;
            this.w2Layered = null;
            this.w3Layered = null;
        }
  
    }

    private static FloatArray loadToContinuesFloatArray(FloatTensor[] input) {
//        System.out.println("Loading to continues float array..." +  input.length + " " + input[0].size());

        int allocationSize = input.length * input[0].size();
        if (allocationSize < 0) {
            throw new IllegalArgumentException("Allocation size is negative: " + allocationSize);
        }

        FloatArray all = new FloatArray(allocationSize);

        int index = 0;
        for (FloatTensor tensor : input) {
            for (int i = 0; i < tensor.size(); i++) {
                all.set(index++, tensor.getFloat(i));
            }
        }

        return all;
    }

    private static FloatArray[] loadToFloatArrayX(FloatTensor[] array) {
        FloatArray[] floatArrays = new FloatArray[array.length];
        for (int i = 0; i < array.length; i++) {
            floatArrays[i] = FloatArray.fromSegment(array[i].asMemorySegment());
            System.out.println("SizeXXX;  " + floatArrays[i].getSize() + " " + array[i].size());
        }
        return floatArrays;
    }

    private static FloatArray[] loadToFloatArray(FloatTensor[] array) {
        FloatArray[] floatArrays = new FloatArray[array.length];
        for (int i = 0; i < array.length; i++) {
            floatArrays[i] = new FloatArray(array[i].size());
            for (int j = 0; j < array[i].size(); j++) {
                floatArrays[i].set(j, array[i].getFloat(j));
            }
//            System.out.println("SizeXXX;  " + floatArrays[i].getSize() + " " + array[i].size());

        }
        return floatArrays;
    }


    private static FloatArray[] loadToFloatArrayX(FloatBuffer[] array) {
        FloatArray[] result = new FloatArray[array.length];
        for (int i = 0; i < array.length; i++) {
            if (array[i] != null) {
                // Create FloatArray with appropriate size
                int size = array[i].remaining();
                result[i] = new FloatArray(size); // Assuming constructor takes size

                // Save current position
                int originalPosition = array[i].position();

                // Copy data
                for (int j = 0; j < size; j++) {
                    result[i].set(j, array[i].get());
                }

                // Optionally reset buffer position
                // array[i].position(originalPosition);
            }
        }
        return result;
    }

    private static FloatArray[] loadToFloatArray(FloatBuffer[] array) {
//        System.out.println("\n=== loadToFloatArray Debug ===");
//        System.out.println("Input array length: " + array.length);

        FloatArray[] result = new FloatArray[array.length];

        for (int i = 0; i < array.length; i++) {
//            System.out.println("\n--- Processing buffer " + i + " ---");

            if (array[i] == null) {
//                System.out.println("Buffer " + i + " is null, skipping");
                result[i] = null;
                continue;
            }

            // Get buffer info
            int size = array[i].remaining();
            int capacity = array[i].capacity();
            int position = array[i].position();
            int limit = array[i].limit();

//            System.out.println("Buffer " + i + " info:");
//            System.out.println("  - capacity: " + capacity);
//            System.out.println("  - position: " + position);
//            System.out.println("  - limit: " + limit);
//            System.out.println("  - remaining: " + size);

            // Create FloatArray with the exact size
            result[i] = new FloatArray(size); // Assuming constructor takes size

            // Save current position for restoration
            int originalPosition = array[i].position();

            // Copy data with verification
//            System.out.println("Copying data...");
            for (int j = 0; j < size; j++) {
                float value = array[i].get();
                result[i].set(j, value);

                // Verbose debug for first few elements
                if (j < 5 || j >= size - 3) {
//                    System.out.printf("  [%d] = %.6f\n", j, value);
                } else if (j == 5) {
//                    System.out.println("  ...");
                }
            }

            // Verify the copy
//            System.out.println("Verifying copy...");
            boolean copyVerified = true;
            array[i].position(originalPosition); // Reset for verification

            for (int j = 0; j < size; j++) {
                float bufferValue = array[i].get();
                float arrayValue = result[i].get(j);

                if (Math.abs(bufferValue - arrayValue) > 0.0001f) {
                    System.err.printf("MISMATCH at index %d: buffer=%.6f, array=%.6f\n",
                            j, bufferValue, arrayValue);
                    copyVerified = false;
                }
            }

            if (copyVerified) {
//                System.out.println("Copy verified successfully!");
            } else {
//                System.err.println("Copy verification FAILED!");
            }

            // Restore buffer position (optional)
            array[i].position(originalPosition);

            // Final stats
//            System.out.println("Final buffer position: " + array[i].position());
//            System.out.println("FloatArray size: " + result[i].getSize()); // Assuming size property exists
        }

//        System.out.println("\n=== loadToFloatArray Complete ===");
        return result;
    }

    private static FloatArray loadToSingleFloatArray(FloatBuffer[] array) {
        int totalSize = 0;
        for (FloatBuffer buffer : array) {
            totalSize += buffer.remaining();
        }

        FloatArray result = new FloatArray(totalSize);
        int index = 0;
        for (FloatBuffer buffer : array) {
            while (buffer.hasRemaining()) {
                result.set(index++, buffer.get());
            }
        }

        return result;
    }

    private static FloatArray loadToSingleFloatArray(FloatBuffer input) {
        FloatBuffer copy = input.duplicate(); // Prevent modifying the original buffer
        int totalSize = copy.remaining();

        FloatArray result = new FloatArray(totalSize);

        int index = 0;
        while (copy.hasRemaining()) {
            result.set(index++, copy.get());
        }

        return result;
    }

    public HalfFloatArray loadToHalfFloatArray(FloatTensor input) {
        HalfFloatArray halfFloatArray = new HalfFloatArray(input.size());

        for (int i = 0; i < input.size(); i++) {
            halfFloatArray.set(i, new HalfFloat(input.getFloat(i)));
        }

        return halfFloatArray;
    }

    public FloatArray loadToFloatArray(FloatTensor input) {
        FloatArray floatArray = new FloatArray(input.size());

        for (int i = 0; i < input.size(); i++) {
            floatArray.set(i, input.getFloat(i));
        }

        return floatArray;
    }

    private VectorFloat4 loadToVectorFloat4Array(FloatTensor[] input) {
        // Calculate total size needed (divide by 4 since each VectorFloat4 holds 4 values)
        int totalElements = 0;
        for (FloatTensor tensor : input) {
            totalElements += tensor.size();
        }

        // Round up to nearest multiple of 4 if necessary
        int vectorSize = (totalElements + 3) / 4;
        VectorFloat4 result = new VectorFloat4(vectorSize);

        int vectorIndex = 0;
        int valueIndex = 0;
        float[] buffer = new float[4];

        for (FloatTensor tensor : input) {
            for (int i = 0; i < tensor.size(); i++) {
                buffer[valueIndex % 4] = tensor.getFloat(i);
                valueIndex++;

                if (valueIndex % 4 == 0) {
                    // We have a complete Float4 vector, add it to the result
                    result.set(vectorIndex++, new Float4(buffer[0], buffer[1], buffer[2], buffer[3]));
                }
            }
        }

        // Handle any remaining values if tensor size wasn't a multiple of 4
        if (valueIndex % 4 != 0) {
            // Fill remaining positions with zeros
            for (int i = valueIndex % 4; i < 4; i++) {
                buffer[i] = 0.0f;
            }
            result.set(vectorIndex, new Float4(buffer[0], buffer[1], buffer[2], buffer[3]));
        }

        return result;
    }

}