package com.example.loader.weights;

import com.example.core.model.tensor.FloatTensor;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.FloatBuffer;
import java.util.Arrays;

public final class WeightsValidator {
    private static final Logger logger = LogManager.getLogger(WeightsValidator.class);
    private final Weights weights;
    private final int dim;
    private final int kvDim;
    private final int hiddenDim;
    private final int numLayers;

    public WeightsValidator(Weights weights, int dim, int kvDim, int hiddenDim, int numLayers) {
        this.weights = weights;
        this.dim = dim;
        this.kvDim = kvDim;
        this.hiddenDim = hiddenDim;
        this.numLayers = numLayers;
    }

    public boolean validateAll() {
        logger.info("Starting full weights validation...");
        printWeightDimensions();
        boolean result = validateTokenEmbeddingTable()
                && validateRmsAttWeights()
                && validateMatmulWeights()
                && validateFfnWeights()
                && validateFinalWeights()
                && validateFreqCisWeights();
        logger.info("Validation completed. Result: {}", result);
        return result;
    }

    public boolean validateTokenEmbeddingTable() {
        logger.info("Validating Token Embedding Table...");
        return validateFloatTensor(weights.token_embedding_table, weights.wcls);
    }

    public boolean validateRmsAttWeights() {
        logger.info("Validating RMS Attention Weights...");
        return validateFloatBufferArray(weights.rms_att_weight, weights.rms_att_weightFlat);
    }

    public boolean validateMatmulWeights() {
        logger.info("Validating MatMul Weights...");
        return validateFloatTensorArray(weights.wq, weights.wqFlat)
                && validateFloatTensorArray(weights.wk, weights.wkFlat)
                && validateFloatTensorArray(weights.wv, weights.wvFlat)
                && validateFloatTensorArray(weights.wo, weights.woFlat);
    }

    public boolean validateFfnWeights() {
        logger.info("Validating FFN Weights...");
        return validateFloatTensorArray(weights.w1, weights.w1Flat)
                && validateFloatTensorArray(weights.w2, weights.w2Flat)
                && validateFloatTensorArray(weights.w3, weights.w3Flat);
    }

    public boolean validateFinalWeights() {
        logger.info("Validating Final Weights...");
        return validateFloatBuffer(weights.rms_final_weight, weights.rms_final_weight_as_floatArray);
    }

    public boolean validateFreqCisWeights() {
        logger.info("Validating Frequency Cis Weights...");
        return validateFloatBuffer(weights.freq_cis_real, weights.freq_cis_realFlat)
                && validateFloatBuffer(weights.freq_cis_imag, weights.freq_cis_imagFlat);
    }

    private boolean validateFloatTensorArray(FloatTensor[] tensors, FloatArray flatArray) {
        logger.debug("Validating FloatTensorArray. Expected size: {}", flatArray.getSize());
        int index = 0;
        for (FloatTensor tensor : tensors) {
            for (int i = 0; i < tensor.size(); i++) {
                if (tensor.getFloat(i) != flatArray.get(index++)) {
                    logger.error("Mismatch found in FloatTensorArray at index {}. Expected: {}, Found: {}", index - 1, tensor.getFloat(i), flatArray.get(index - 1));
                    return false;
                }
            }
        }
        return true;
    }

    private boolean validateFloatTensor(FloatTensor tensor, FloatTensor flatTensor) {
        logger.debug("Validating FloatTensor. Size: {}", tensor.size());
        for (int i = 0; i < tensor.size(); i++) {
            if (tensor.getFloat(i) != flatTensor.getFloat(i)) {
                logger.error("Mismatch found in FloatTensor at index {}. Expected: {}, Found: {}", i, tensor.getFloat(i), flatTensor.getFloat(i));
                return false;
            }
        }
        return true;
    }

    private boolean validateFloatBuffer(FloatBuffer buffer, FloatArray flatArray) {
        logger.debug("Validating FloatBuffer. Expected size: {}, Actual size: {}", flatArray.getSize(), buffer.remaining());
        if (buffer.remaining() != flatArray.getSize()) {
            logger.error("Buffer size mismatch. Expected: {}, Actual: {}", flatArray.getSize(), buffer.remaining());
            return false;
        }

        float[] bufferData = new float[buffer.remaining()];
        buffer.get(bufferData);
        if (!Arrays.equals(bufferData, flatArray.toHeapArray())) {
            logger.error("Mismatch found in FloatBuffer validation.");
            return false;
        }
        return true;
    }

    private boolean validateFloatBufferArray(FloatBuffer[] buffers, FloatArray flatArray) {
        logger.debug("Validating FloatBufferArray. Expected size: {}", flatArray.getSize());
        int index = 0;
        for (FloatBuffer buffer : buffers) {
            FloatBuffer copy = buffer.duplicate();
            logger.debug("Processing buffer with remaining size: {}", copy.remaining());
            float[] bufferData = new float[copy.remaining()];
            copy.get(bufferData);

            for (float value : bufferData) {
                if (value != flatArray.get(index++)) {
                    logger.error("Mismatch found in FloatBufferArray at index {}. Expected: {}, Found: {}", index - 1, value, flatArray.get(index - 1));
                    return false;
                }
            }
        }
        return true;
    }

    public void printWeightDimensions() {
        System.out.println("\n==== WEIGHT DIMENSIONS ====");
        System.out.println("Model configuration:");
        System.out.println("  Number of layers: " + numLayers);
        System.out.println("  Dimension (dim): " + dim);
        System.out.println("  KV Dimension (kvDim): " + kvDim);
        System.out.println("  Hidden Dimension (hiddenDim): " + hiddenDim);

        // Print expected sizes
        System.out.println("\nExpected dimensions per layer:");
        System.out.println("  wq: dim × dim = " + dim + " × " + dim + " = " + (dim * dim));
        System.out.println("  wk: dim × kvDim = " + dim + " × " + kvDim + " = " + (dim * kvDim));
        System.out.println("  wv: dim × kvDim = " + dim + " × " + kvDim + " = " + (dim * kvDim));
        System.out.println("  wo: dim × dim = " + dim + " × " + dim + " = " + (dim * dim));
        System.out.println("  w1: dim × hiddenDim = " + dim + " × " + hiddenDim + " = " + (dim * hiddenDim));
        System.out.println("  w2: hiddenDim × dim = " + hiddenDim + " × " + dim + " = " + (hiddenDim * dim));
        System.out.println("  w3: dim × hiddenDim = " + dim + " × " + hiddenDim + " = " + (dim * hiddenDim));

        System.out.println("\nExpected total sizes (all layers):");
        System.out.println("  wq: numLayers × dim × dim = " + numLayers + " × " + (dim * dim) + " = " + (numLayers * dim * dim));
        System.out.println("  wk: numLayers × dim × kvDim = " + numLayers + " × " + (dim * kvDim) + " = " + (numLayers * dim * kvDim));
        System.out.println("  wv: numLayers × dim × kvDim = " + numLayers + " × " + (dim * kvDim) + " = " + (numLayers * dim * kvDim));
        System.out.println("  wo: numLayers × dim × dim = " + numLayers + " × " + (dim * dim) + " = " + (numLayers * dim * dim));

        // Print actual sizes from individual tensors
        System.out.println("\nActual tensor dimensions:");
        int totalWqSize = 0;
        int totalWkSize = 0;
        int totalWvSize = 0;
        int totalWoSize = 0;

        for (int l = 0; l < numLayers; l++) {
            int wqSize = weights.wq[l].size();
            int wkSize = weights.wk[l].size();
            int wvSize = weights.wv[l].size();
            int woSize = weights.wo[l].size();

            totalWqSize += wqSize;
            totalWkSize += wkSize;
            totalWvSize += wvSize;
            totalWoSize += woSize;

            if (l < 2) { // Just print first two layers to avoid output flooding
                System.out.println("Layer " + l + ":");
                System.out.println("  wq[" + l + "].size() = " + wqSize + " (expected: " + (dim * dim) + ")");
                System.out.println("  wk[" + l + "].size() = " + wkSize + " (expected: " + (dim * kvDim) + ")");
                System.out.println("  wv[" + l + "].size() = " + wvSize + " (expected: " + (dim * kvDim) + ")");
                System.out.println("  wo[" + l + "].size() = " + woSize + " (expected: " + (dim * dim) + ")");
            }
        }

        System.out.println("\nTotal tensor sizes (all layers):");
        System.out.println("  Sum of all wq sizes: " + totalWqSize + " (expected: " + (numLayers * dim * dim) + ")");
        System.out.println("  Sum of all wk sizes: " + totalWkSize + " (expected: " + (numLayers * dim * kvDim) + ")");
        System.out.println("  Sum of all wv sizes: " + totalWvSize + " (expected: " + (numLayers * dim * kvDim) + ")");
        System.out.println("  Sum of all wo sizes: " + totalWoSize + " (expected: " + (numLayers * dim * dim) + ")");

        // Print sizes of flattened arrays
        System.out.println("\nFlattened array dimensions:");
        System.out.println("  wqFlat.getSize() = " + weights.wqFlat.getSize() + " (expected: " + (numLayers * dim * dim) + ")");
        System.out.println("  wkFlat.getSize() = " + weights.wkFlat.getSize() + " (expected: " + (numLayers * dim * kvDim) + ")");
        System.out.println("  wvFlat.getSize() = " + weights.wvFlat.getSize() + " (expected: " + (numLayers * dim * kvDim) + ")");
        System.out.println("  woFlat.getSize() = " + weights.woFlat.getSize() + " (expected: " + (numLayers * dim * dim) + ")");

        // Print weight layout inspection for matrix multiplication validation
        printWeightLayoutInspection();

        System.out.println("\n==== END WEIGHT DIMENSIONS ====\n");
    }

    private void printWeightLayoutInspection() {
        System.out.println("\nWeight Layout Inspection (for matrix multiplication validation):");

        // Check layer offsets for wk and wv matrices (the ones having issues)
        for (int l = 0; l < Math.min(numLayers, 2); l++) {
            // Expected offsets
            int wqOffset = l * dim * dim;
            int wkOffset = l * dim * kvDim;
            int wvOffset = l * dim * kvDim;

            System.out.println("Layer " + l + " Weight Sample Values:");

            // Print first few values from wq, wk, wv with offsets
            System.out.println("  wqFlat (layer offset = " + wqOffset + "):");
            for (int i = 0; i < 5; i++) {
                System.out.println("    wqFlat[" + (wqOffset + i) + "] = " + weights.wqFlat.get(wqOffset + i));
            }

            System.out.println("  wkFlat (layer offset = " + wkOffset + "):");
            for (int i = 0; i < 5; i++) {
                System.out.println("    wkFlat[" + (wkOffset + i) + "] = " + weights.wkFlat.get(wkOffset + i));
            }

            System.out.println("  wvFlat (layer offset = " + wvOffset + "):");
            for (int i = 0; i < 5; i++) {
                System.out.println("    wvFlat[" + (wvOffset + i) + "] = " + weights.wvFlat.get(wvOffset + i));
            }

            // Check specific matrix multiplication indices for first few elements
            System.out.println("\n  Matrix Multiplication Index Tests for Layer " + l + ":");
            for (int i = 0; i < 2; i++) { // First 2 rows
                for (int j = 0; j < 3; j++) { // First 3 columns
                    int wqIndex = wqOffset + i * dim + j;
                    int wkIndex = wkOffset + i * dim + j;
                    int wvIndex = wvOffset + i * dim + j;

                    System.out.println("    wq[row=" + i + ", col=" + j + "] = wqFlat[" + wqIndex + "] = " +
                            weights.wqFlat.get(wqIndex));

                    if (j < kvDim) {
                        System.out.println("    wk[row=" + i + ", col=" + j + "] = wkFlat[" + wkIndex + "] = " +
                                weights.wkFlat.get(wkIndex));
                        System.out.println("    wv[row=" + i + ", col=" + j + "] = wvFlat[" + wvIndex + "] = " +
                                weights.wvFlat.get(wvIndex));
                    }
                }
            }
        }
    }

}
