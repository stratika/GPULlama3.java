package com.example.inference.weights.tornado;

import com.example.core.model.GGMLType;
import com.example.inference.weights.standard.StandardWeights;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * A model-specific implementation of {@link TornadoWeights} for the Llama model.
 * This class encapsulates the weights required for performing GPU-accelerated
 * inference of the Llama model using TornadoVM.
 *
 * <p><b>Note:</b> This weight format can also be used with the Mistral model.</p>
 */
public class LlamaTornadoWeights extends TornadoWeights {

    // @formatter:off
    public LlamaTornadoWeights(
            FloatArray tokenEmbeddingTable,
            FloatArray[] rms_att_weightLayered,
            HalfFloatArray[] wqLayered,
            HalfFloatArray[] wkLayered,
            HalfFloatArray[] wvLayered,
            HalfFloatArray[] woLayered,
            FloatArray[] rms_ffn_weightLayered,
            HalfFloatArray[] w1Layered,
            HalfFloatArray[] w2Layered,
            HalfFloatArray[] w3Layered,
            FloatArray rms_final_weight_as_floatArray,
            FloatArray freq_cis_realFlat,
            FloatArray freq_cis_imagFlat,
            HalfFloatArray wclsByteArray,
            GGMLType weightType) {
        // call to TornadoWeights constructor
        super(tokenEmbeddingTable,
                rms_att_weightLayered,
                wqLayered,
                wkLayered,
                wvLayered,
                woLayered,
                rms_ffn_weightLayered,
                w1Layered,
                w2Layered,
                w3Layered,
                rms_final_weight_as_floatArray,
                freq_cis_realFlat,
                freq_cis_imagFlat,
                wclsByteArray,
                weightType);
    }
    // @formatter:on
}
