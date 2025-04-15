package com.example.tornadovm;

import com.example.inference.engine.impl.Configuration;
import com.example.inference.engine.impl.Llama;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

/**
 * Enhanced TornadoVMMasterPlan with detailed per-layer debugging
 */
public class TornadoVMMasterPlanDebug {
    private final State state;
    private final Configuration config;
    private final Weights weights;
    private final List<ImmutableTaskGraph> taskGraphs;
    private final GridScheduler scheduler;
    private TornadoExecutionPlan executionPlan;

    // Debug settings
    private static final boolean DEBUG_ENABLED = Boolean.parseBoolean(System.getProperty("tornado.layerDebug", "false"));
    private static final String DEBUG_LOG_FILE = "tornado_layer_debug.log";
    private static PrintWriter debugLogWriter;

    static {
        if (DEBUG_ENABLED) {
            try {
                File logFile = new File(DEBUG_LOG_FILE);
                if (logFile.exists()) {
                    logFile.delete();
                }
                debugLogWriter = new PrintWriter(new FileWriter(DEBUG_LOG_FILE, true));
                debugLogWriter.println("===== TornadoVM Layer-by-Layer Debug Log =====");
                debugLogWriter.flush();
            } catch (IOException e) {
                System.err.println("Failed to initialize TornadoVM debug log: " + e.getMessage());
            }
        }
    }

    public TornadoVMMasterPlanDebug(State state, Llama model) {
        TornadoVMLayerPlanner tornadoVMLayerPlanner = new TornadoVMLayerPlanner(state, model);
        var tornadoVMPlan = tornadoVMLayerPlanner.setupTornadoForwardPlan();
        this.taskGraphs = tornadoVMPlan.getFirst();
        this.scheduler = tornadoVMPlan.getSecond();
        this.state = state;
        this.config = model.configuration();
        this.weights = model.weights();
        this.executionPlan = new TornadoExecutionPlan(taskGraphs.toArray(new ImmutableTaskGraph[taskGraphs.size()]));

        if (DEBUG_ENABLED) {
            logDebug("TornadoVMMasterPlanDebug initialized");
            logDebug("Number of task graphs: " + taskGraphs.size());
            logDebug("Model configuration:");
            logDebug("  dim: " + config.dim);
            logDebug("  headSize: " + config.headSize);
            logDebug("  numberOfHeads: " + config.numberOfHeads);
            logDebug("  numberOfKeyValueHeads: " + config.numberOfKeyValueHeads);
            logDebug("  numberOfLayers: " + config.numberOfLayers);
            logDebug("  contextLength: " + config.contextLength);
        }
    }

    /**
     * Enhanced forward execution with detailed per-layer debugging
     */
    public void tornadoVMForwardExecute(int position) {
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;

        state.positionAndLayer.set(0, position);
        logDebug("Starting forward execution for position " + position);

        // Update before execute (it an every copy in)
        logDebug("Executing initial graph (0) for buffer setup");
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        // Check embedding values
        logDebug("After initial execution:");
        checkTensorState("wrapX", state.wrapX, 10);

        for (int l = 0; l < config.numberOfLayers; l++) {
            logDebug("\n=== Processing layer " + l + " ===");
            state.positionAndLayer.set(1, l);

            int loff = l * config.contextLength * kvDim;
            state.positionAndLayer.set(3, loff);
            int layerOffsetForCaches = loff + position * kvDim;
            state.positionAndLayer.set(2, layerOffsetForCaches);

            logDebug("Layer offsets: loff=" + loff + ", cacheOffset=" + layerOffsetForCaches);

            // Layer taskgraph
            logDebug("Executing layer graph (1) for layer " + l);
            TornadoExecutionResult result = executionPlan.withGraph(1).withGridScheduler(scheduler).execute();
            logDebug("Layer execution completed with status: " + result.isReady());

            // Check intermediate states after each layer
            logDebug("After layer " + l + " execution:");
            checkTensorState("wrapX", state.wrapX, 10);
            checkTensorState("wrapXb", state.wrapXb, 10);
            checkTensorState("wrapQ", state.wrapQ, 10);
            checkTensorState("wrapK", state.wrapK, 10);
            checkTensorState("wrapV", state.wrapV, 10);

            // Debug statistics
            if (DEBUG_ENABLED) {
                logMemoryUsage(result, "Layer " + l, l);
            }
        }

        // Final RMSNorm and Logits
        logDebug("\n=== Executing final RMSNorm and logits projection ===");
        TornadoExecutionResult result = executionPlan.withGraph(2).withGridScheduler(scheduler).execute();
        logDebug("Final execution completed with status: " + result.isReady());

        // Verify logits output
        logDebug("Final logits state:");
        checkTensorState("wrapLogits", state.wrapLogits, 20);

        // Check if all values in logits are zero
        boolean allZeros = true;
        float sum = 0.0f;
        float[] logitsArray = state.wrapLogits.toHeapArray();
        for (int i = 0; i < logitsArray.length; i++) {
            sum += Math.abs(logitsArray[i]);
            if (Math.abs(logitsArray[i]) > 1e-6) {
                allZeros = false;
            }
        }
        logDebug("Logits all zeros? " + allZeros);
        logDebug("Sum of absolute values in logits: " + sum);

        // Debug statistics
        if (DEBUG_ENABLED) {
            logMemoryUsage(result, "Final RMSNorm and Logits", -1);
        }

        // Now check if data is actually transferred to the FloatTensor
        logDebug("Checking transfer from TornadoVM buffer to FloatTensor:");
        float[] cpuLogits = new float[Math.min(10, state.logits.size())];
        for (int i = 0; i < cpuLogits.length; i++) {
            cpuLogits[i] = state.logits.getFloat(i);
        }
        logDebug("CPU logits (first few): " + formatFloatArray(cpuLogits));
    }

    /**
     * Check and log the state of a TornadoVM tensor
     */
    private void checkTensorState(String name, uk.ac.manchester.tornado.api.types.arrays.FloatArray tensor, int sampleSize) {
        if (!DEBUG_ENABLED || tensor == null) return;

        int size = tensor.getSize();
        float[] sample = new float[Math.min(sampleSize, size)];

        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        float sum = 0.0f;
        int nanCount = 0;
        int infCount = 0;
        int zeroCount = 0;

        for (int i = 0; i < size; i++) {
            float val = tensor.get(i);

            if (i < sample.length) {
                sample[i] = val;
            }

            if (Float.isNaN(val)) {
                nanCount++;
                continue;
            }

            if (Float.isInfinite(val)) {
                infCount++;
                continue;
            }

            if (val == 0.0f) {
                zeroCount++;
            }

            min = Math.min(min, val);
            max = Math.max(max, val);
            sum += val;
        }

        float mean = (size - nanCount - infCount > 0) ? sum / (size - nanCount - infCount) : 0.0f;

        logDebug(name + " stats:");
        logDebug("  Size: " + size);
        logDebug("  Range: [" + min + ", " + max + "]");
        logDebug("  Mean: " + mean);
        logDebug("  NaN count: " + nanCount);
        logDebug("  Infinity count: " + infCount);
        logDebug("  Zero count: " + zeroCount + " (" + (100.0f * zeroCount / size) + "%)");
        logDebug("  Sample values: " + formatFloatArray(sample));
    }

    private String formatFloatArray(float[] array) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < array.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.6f", array[i]));
        }
        return sb.toString();
    }

    private void logMemoryUsage(TornadoExecutionResult executionResult, String stage, int layer) {
        try {
            long totalDeviceMemoryUsage = executionResult.getProfilerResult().getTotalDeviceMemoryUsage();
            double memoryInMB = totalDeviceMemoryUsage / (1024.0 * 1024.0);
            String message;
            if (layer >= 0) {
                message = String.format("Layer %d, %s: Total memory usage = %.2f MB", layer, stage, memoryInMB);
            } else {
                message = String.format("%s: Total memory usage = %.2f MB", stage, memoryInMB);
            }
            logDebug(message);
        } catch (Exception e) {
            logDebug("Error getting memory usage: " + e.getMessage());
        }
    }

    private static void logDebug(String message) {
        if (!DEBUG_ENABLED || debugLogWriter == null) return;

        debugLogWriter.println(message);
        debugLogWriter.flush();

        System.err.println("[TornadoDebug] " + message);
    }

    /**
     * Release resources
     */
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();

        if (debugLogWriter != null) {
            debugLogWriter.close();
        }
    }
}