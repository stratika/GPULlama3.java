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
        return validateTokenEmbeddingTable()
                && validateRmsAttWeights()
                && validateMatmulWeights()
                && validateFfnWeights()
                && validateFinalWeights()
                && validateFreqCisWeights();
    }

    public boolean validateTokenEmbeddingTable() {
        return validateFloatTensor(weights.token_embedding_table, weights.wcls);
    }

    public boolean validateRmsAttWeights() {
        return validateFloatBufferArray(weights.rms_att_weight, weights.rms_att_weightFlat);
    }

    public boolean validateMatmulWeights() {
        return validateFloatTensorArray(weights.wq, weights.wqFlat)
                && validateFloatTensorArray(weights.wk, weights.wkFlat)
                && validateFloatTensorArray(weights.wv, weights.wvFlat)
                && validateFloatTensorArray(weights.wo, weights.woFlat);
    }

    public boolean validateFfnWeights() {
        return validateFloatTensorArray(weights.w1, weights.w1Flat)
                && validateFloatTensorArray(weights.w2, weights.w2Flat)
                && validateFloatTensorArray(weights.w3, weights.w3Flat);
    }

    public boolean validateFinalWeights() {
        return validateFloatBuffer(weights.rms_final_weight, weights.rms_final_weight_as_floatArray);
    }

    public boolean validateFreqCisWeights() {
        return validateFloatBuffer(weights.freq_cis_real, weights.freq_cis_realFlat)
                && validateFloatBuffer(weights.freq_cis_imag, weights.freq_cis_imagFlat);
    }

    private boolean validateFloatTensorArray(FloatTensor[] tensors, FloatArray flatArray) {
        int index = 0;
        for (FloatTensor tensor : tensors) {
            for (int i = 0; i < tensor.size(); i++) {
                if (tensor.getFloat(i) != flatArray.get(index++)) {
                    logger.error("Mismatch found in FloatTensorArray at index {}", index - 1);
                    return false;
                }
            }
        }
        return true;
    }

    private boolean validateFloatTensor(FloatTensor tensor, FloatTensor flatTensor) {
        for (int i = 0; i < tensor.size(); i++) {
            if (tensor.getFloat(i) != flatTensor.getFloat(i)) {
                logger.error("Mismatch found in FloatTensor at index {}", i);
                return false;
            }
        }
        return true;
    }

    private boolean validateFloatBuffer(FloatBuffer buffer, FloatArray flatArray) {
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
        int index = 0;
        for (FloatBuffer buffer : buffers) {
            // Duplicate buffer to avoid modifying original position
            FloatBuffer copy = buffer.duplicate();

            float[] bufferData = new float[copy.remaining()];
            copy.get(bufferData); // Read without modifying the original buffer

            for (float value : bufferData) {
                if (value != flatArray.get(index++)) {
                    logger.error("Mismatch found in FloatBufferArray at index {}", index - 1);
                    return false;
                }
            }
        }
        return true;
    }
}