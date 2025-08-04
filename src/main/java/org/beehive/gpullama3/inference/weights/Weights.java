package org.beehive.gpullama3.inference.weights;

import org.beehive.gpullama3.core.model.GGMLType;

/**
 * The GPULlama3.java utilizes two distinct weight types:
 * <ul>
 *   <li><b>StandardWeights:</b> Designed for standard Java-based inference on the CPU.</li>
 *   <li><b>TornadoWeights:</b> Optimized for GPU-accelerated inference using TornadoVM.</li>
 * </ul>
 *
 * The packages <code>weights.standard</code> and <code>weights.tornado</code> define
 * base classes and model-specific implementations for weights in their respective formats.
 */
public interface Weights {

    GGMLType getWeightType();

}