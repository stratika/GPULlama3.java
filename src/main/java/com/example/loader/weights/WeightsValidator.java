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

    public WeightsValidator(Weights weights) {
        this.weights = weights;
    }

    public boolean validateAll() {
        logger.info("Starting full weights validation...");
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
}
