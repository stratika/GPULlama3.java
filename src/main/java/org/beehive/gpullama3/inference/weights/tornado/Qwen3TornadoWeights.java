package org.beehive.gpullama3.inference.weights.tornado;

import org.beehive.gpullama3.core.model.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * A model-specific implementation of {@link TornadoWeights} for the Qwen3 model.
 * This class encapsulates the weights required for performing GPU-accelerated
 * inference of the Qwen3 model using TornadoVM.
 *
 * <p><b>Note:</b> This weight format can also be used with the Mistral model.</p>
 */
public class Qwen3TornadoWeights extends TornadoWeights {

    //attnKNorm
    public FloatArray[] rms_att_KNormLayered;
    //attnQNorm
    public FloatArray[] rms_att_QNormLayered;

    // @formatter:off
    public Qwen3TornadoWeights(
            FloatArray tokenEmbeddingTable,
            FloatArray[] rms_att_weightLayered,
            HalfFloatArray[] wqLayered,
            HalfFloatArray[] wkLayered,
            HalfFloatArray[] wvLayered,
            HalfFloatArray[] woLayered,
            FloatArray[] rms_att_KNormLayered,
            FloatArray[] rms_att_QNormLayered,
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
        // init qwen3-specific fields
        this.rms_att_KNormLayered = rms_att_KNormLayered;
        this.rms_att_QNormLayered = rms_att_QNormLayered;
    }
    // @formatter:on

}
