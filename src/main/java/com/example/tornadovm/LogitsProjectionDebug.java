//package com.example.tornadovm;
//
//import com.example.core.model.tensor.FloatTensor;
//import com.example.inference.engine.impl.Llama;
//import com.example.loader.weights.ModelLoader;
//import com.example.loader.weights.State;
//import com.example.loader.weights.Weights;
//import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
//import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
//import uk.ac.manchester.tornado.api.KernelContext;
//
//import java.io.FileWriter;
//import java.io.IOException;
//import java.io.PrintWriter;
//import java.nio.file.Path;
//
///**
// * Debug utility specifically for the final projection to logits in TornadoVM
// */
//public class LogitsProjectionDebug {
//    public static void main(String[] args) throws IOException {
//        if (args.length < 1) {
//            System.err.println("Usage: LogitsProjectionDebug <model_path>");
//            System.exit(1);
//        }
//
//        String modelPath = args[0];
//        PrintWriter logWriter = new PrintWriter(new FileWriter("logits_projection_debug.log"));
//
//        try {
//            System.out.println("Loading model: " + modelPath);
//            Llama model = ModelLoader.loadModel(Path.of(modelPath), 512, true);
//
//            // Get configuration and weights
//            var config = model.configuration();
//            Weights weights = model.weights();
//
//            // Create state with simple test data
//            State state = new State(config);
//
//            // Initialize X with non-zero values
//            for (int i = 0; i < config.dim; i++) {
//                state.x.setFloat(i, 0.1f * (i % 10 + 1));
//                state.wrapX.set(i, 0.1f * (i % 10 + 1));
//            }
//
//            // Log some properties of the weights
//            logWriter.println("Model configuration:");
//            logWriter.println("  Vocabulary size: " + config.vocabularySize);
//            logWriter.println("  Dimension: " + config.dim);
//
//            // Check wcls (the weight matrix for projecting to logits)
//            logWriter.println("\nChecking weight matrix (wcls):");
//            logWriter.println("  Type: " + weights.wcls.type());
//            logWriter.println("  Size: " + weights.wcls.size());
//
//            // Check the ByteArray for TornadoVM
//            logWriter.println("\nChecking TornadoVM ByteArray (wclsByteArray):");
//            logWriter.println("  Size: " + weights.wclsByteArray.getSize());
//
//            // Check first few bytes
//            logWriter.println("  First 20 bytes:");
//            for (int i = 0; i < Math.min(20, weights.wclsByteArray.getSize()); i++) {
//                logWriter.print(String.format("%02X ", weights.wclsByteArray.get(i) & 0xFF));
//                if ((i + 1) % 8 == 0) logWriter.println();
//            }
//            logWriter.println();
//
//            // Create sample X data
//            logWriter.println("\nSample X data (first 10 values):");
//            for (int i = 0; i < 10; i++) {
//                logWriter.printf("  x[%d] = %.6f%n", i, state.x.getFloat(i));
//            }
//
//            // Run CPU projection as reference
//            logWriter.println("\nRunning CPU projection:");
//            weights.wcls.matmul(state.x, state.logits, config.vocabularySize, config.dim);
//
//            logWriter.println("CPU logits (first 20 values):");
//            float cpuSum = 0.0f;
//            for (int i = 0; i < Math.min(20, state.logits.size()); i++) {
//                float val = state.logits.getFloat(i);
//                logWriter.printf("  logits[%d] = %.6f%n", i, val);
//                cpuSum += Math.abs(val);
//            }
//            logWriter.println("CPU sum of absolute values (first 20): " + cpuSum);
//
//            // Manually test the TornadoVM implementation
//            logWriter.println("\nTesting TornadoVM implementation manually:");
//            // Reset logits
//            for (int i = 0; i < state.wrapLogits.getSize(); i++) {
//                state.wrapLogits.set(i, 0.0f);
//            }
//
//            // Create a dummy kernel context for testing
//            KernelContext context = new KernelContext();
//            context = 0;
//            context.localIdx = 0;
//
//            // Test the implementation directly
//            logWriter.println("Testing matmulTornadoQ8 with first few indices:");
//            for (int idx = 0; idx < Math.min(20, config.vocabularySize); idx++) {
//                context.globalIdx = idx;
//                float result = computeMatmulQ8(context, weights.wclsByteArray, state.wrapX, idx, config.dim);
//                state.wrapLogits.set(idx, result);
//                logWriter.printf("  idx=%d, result=%.6f%n", idx, result);
//            }
//
//            // Check results
//            logWriter.println("\nTornadoVM logits (first 20 values):");
//            float tornadoSum = 0.0f;
//            for (int i = 0; i < Math.min(20, state.wrapLogits.getSize()); i++) {
//                float val = state.wrapLogits.get(i);
//                logWriter.printf("  logits[%d] = %.6f%n", i, val);
//                tornadoSum += Math.abs(val);
//            }
//            logWriter.println("TornadoVM sum of absolute values (first 20): " + tornadoSum);
//
//            // Copy from TornadoVM buffer to state logits for comparison
//            logWriter.println("\nCopying from TornadoVM buffer to state logits:");
//            for (int i = 0; i < Math.min(state.logits.size(), state.wrapLogits.getSize()); i++) {
//                state.logits.setFloat(i, state.wrapLogits.get(i));
//            }
//
//            // Compare CPU and TornadoVM results
//            logWriter.println("\nComparison of CPU and TornadoVM results:");
//            float maxDiff = 0.0f;
//            int maxDiffIdx = -1;
//            for (int i = 0; i < Math.min(20, state.logits.size()); i++) {
//                float cpuVal = state.logits.getFloat(i);
//                float tornadoVal = state.wrapLogits.get(i);
//                float diff = Math.abs(cpuVal - tornadoVal);
//                logWriter.printf("  idx=%d, CPU=%.6f, TornadoVM=%.6f, diff=%.6f%n",
//                        i, cpuVal, tornadoVal, diff);
//                if (diff > maxDiff) {
//                    maxDiff = diff;
//                    maxDiffIdx = i;
//                }
//            }
//            logWriter.println("Maximum difference: " + maxDiff + " at index " + maxDiffIdx);
//
//            System.out.println("Debug completed. Check logits_projection_debug.log for details.");
//
//        } finally {
//            logWriter.close();
//        }
//    }
//
//    /**
//     * Implementation of matmulTornadoQ8 for testing
//     */
//    private static float computeMatmulQ8(KernelContext context, ByteArray thisx, FloatArray that, int idx, int dim1) {
//        final int BLOCK_SIZE = 32; // Q8 block size
//        final int BYTES_PER_BLOCK = 2 + BLOCK_SIZE; // 2 bytes for scale + 32 bytes for values
//
//        float result = 0f;
//        int thisOffset = idx * dim1;
//
//        for (int j = 0; j < dim1; j++) {
//            int index = thisOffset + j;
//
//            // Calculate block position
//            int blockIndex = index / BLOCK_SIZE;
//            int withinBlockIndex = index % BLOCK_SIZE;
//            int blockOffset = blockIndex * BYTES_PER_BLOCK;
//
//            // Read scale (float16) for this block
//            int scaleByte1 = thisx.get(blockOffset) & 0xFF;
//            int scaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
//            short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
//            float scale = decodeFloat16(scaleFloat16);
//
//            // Read quantized value
//            byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);
//
//            // Dequantize and multiply
//            result += (quantized * scale) * that.get(j);
//        }
//
//        return result;
//    }
//
//    /**
//     * Decode a float16 value to float32
//     */
//    private static float decodeFloat16(short value) {
//        int sign = (value & 0x8000) >>> 15;
//        int exp = (value & 0x7C00) >>> 10;
//        int frac = value & 0x03FF;
//
//        // Handle special cases
//        if (exp == 0x1F) {
//            return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
//        }
//
//        if (exp == 0) {
//            if (frac == 0) {
//                return sign == 0 ? 0.0f : -0.0f;
//            }
//            // Denormalized
//            float result = frac * pow2(-24);
//            return sign == 0 ? result : -result;
//        }
//
//        // Normalized
//        float result = 1.0f + (frac / 1024.0f);
//        result *= pow2(exp - 15);
//        return sign == 0 ? result : -result;
//    }
//
//    /**
//     * Compute 2^n efficiently
//     */
//    private static float pow2(int n) {
//        if (n >= 0) {
//            if (n < 31) {
//                return (float) (1 << n);
//            }
//            return Float.POSITIVE_INFINITY;
//        }
//        if (n > -150) {
//            return 1.0f / (1 << -n);
//        }
//        return 0.0f;
//    }
//}