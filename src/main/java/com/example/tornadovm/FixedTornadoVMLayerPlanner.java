package com.example.tornadovm;

import com.example.aux.Tuple2;
import com.example.inference.engine.impl.Configuration;
import com.example.inference.engine.impl.Llama;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import uk.ac.manchester.tornado.api.AccessorParameters;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Fixed version of TornadoVMLayerPlanner with debugging for the logits projection
 */
public class FixedTornadoVMLayerPlanner {

    private final State state;
    private final Configuration config;
    private final Weights weights;
    private static final boolean DEBUG_MODE = Boolean.parseBoolean(System.getProperty("tornado.debug.logits", "false"));

    public FixedTornadoVMLayerPlanner(State state, Llama model) {
        this.state = state;
        this.config = model.configuration();
        this.weights = model.weights();
    }

    private GridScheduler setupGridSchedulers() {
        GridScheduler tornadoForwardScheduler = new GridScheduler();

        // Create common worker grids that will be used across different schedulers
        WorkerGrid dimWorker = new WorkerGrid1D(config.dim);
        dimWorker.setGlobalWork(config.dim, 1, 1);
        dimWorker.setLocalWork(256, 1, 1);

        WorkerGrid headsWorker = new WorkerGrid1D(config.numberOfHeads * 64);
        headsWorker.setGlobalWork(config.numberOfHeads * 64, 1, 1);
        headsWorker.setLocalWork(64, 1, 1);

        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        WorkerGrid hiddenDimWorker = new WorkerGrid1D(config.hiddenDim);
        hiddenDimWorker.setGlobalWork(config.hiddenDim, 1, 1);
        hiddenDimWorker.setLocalWork(256, 1, 1);

        // For the vocabulary projection, ensure we have enough parallelism
        // but don't exceed hardware limits
        int blockSize = 256;
        int numBlocks = Math.min(256, (config.vocabularySize + blockSize - 1) / blockSize);
        WorkerGrid vocabWorker = new WorkerGrid1D(config.vocabularySize);
        vocabWorker.setGlobalWork(config.vocabularySize, 1, 1);
        vocabWorker.setLocalWork(blockSize, 1, 1);

        // Create a 2D worker grid for the final projection matrix multiply
        // This can significantly improve performance for large matrices
        int BLOCK_SIZE = 16;
        int blockWidth = (config.vocabularySize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blockHeight = (config.dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        WorkerGrid matmulWorker = new WorkerGrid2D(blockWidth, blockHeight);
        matmulWorker.setGlobalWork(blockWidth * BLOCK_SIZE, blockHeight * BLOCK_SIZE, 1);
        matmulWorker.setLocalWork(BLOCK_SIZE, BLOCK_SIZE, 1);

        WorkerGrid ropeWorker = new WorkerGrid1D(config.dim / 2);
        ropeWorker.setGlobalWork(config.dim / 2, 1, 1);
        ropeWorker.setLocalWork(128, 1, 1);

        // Create a scheduler for each task graph
        tornadoForwardScheduler.addWorkerGrid("updX.copyinX", singleWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.reductionOneBlock", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.normalize1", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.rope", ropeWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.residual1", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.reductionOneBlockFFN", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.normalizeFNN", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.residual2", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("rms_logits.reductionOneBlock", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("rms_logits.normalize", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("rms_logits.projection", vocabWorker);
        tornadoForwardScheduler.addWorkerGrid("rms_logits.check", singleWorker);

        return tornadoForwardScheduler;
    }

    /**
     * Sets up the forward plan for TornadoVM execution with improved logits projection
     */
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlan() {
        List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

        int dim = config.dim;
        int headSize = config.headSize;
        int numHeads = config.numberOfHeads;
        int numKVHeads = config.numberOfKeyValueHeads;
        int kvDim = (dim * numKVHeads) / numHeads;
        int kvMul = numHeads / numKVHeads;

        // Define worker grid sizes
        int localSizeRMS = 256;
        int localSizeHeads = 64;

        int reduceArraySize = validateAndAdjustBufferSizes();

        FloatArray intermediateReduceFirst = new FloatArray(reduceArraySize);
        FloatArray intermediateReduceTwo = new FloatArray(reduceArraySize);
        FloatArray intermediateReduceThree = new FloatArray(reduceArraySize);

        intermediateReduceFirst.init(0.0f);
        intermediateReduceTwo.init(0.0f);
        intermediateReduceThree.init(0.0f);
        FloatArray maxValues = new FloatArray(1);
        FloatArray expValues = new FloatArray(headSize);
        FloatArray sumValues = new FloatArray(1);

        // Debug buffer for tracking issues
        FloatArray debugBuffer = new FloatArray(10);
        debugBuffer.init(0.0f);

        // Create kernel context
        KernelContext context = new KernelContext();

        // ================ TASK GRAPH 0: BUFFER INITIALIZATION ================
        TaskGraph updX = new TaskGraph("updX")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, state.wrapX)
                .task("copyinX", TornadoVMCompute::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX);
        taskGraphs.add(updX.snapshot());

        // =====================================================================================
        TaskGraph unifiedLayer = new TaskGraph("layer")
                // Initial data transfer
                .consumeFromDevice(state.wrapX)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        // RMS Norm weights and buffers
                        weights.rms_att_weightFlat, intermediateReduceFirst,
                        // QKV weights
                        weights.wqFlat, weights.wkFlat, weights.wvFlat,
                        // FFN weights and buffers
                        weights.woFlat, weights.rms_ffn_weightFlat,
                        weights.w1Flat, weights.w2Flat, weights.w3Flat,
                        intermediateReduceTwo, state.wrapHb, state.wrapHb2, state.wrapXb2,
                        // Attention buffers
                        state.wrapAtt, maxValues, expValues, sumValues,
                        // Caches
                        state.wrapKeyCache, state.wrapValueCache, context
                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionAndLayer)

                // -------- RMS NORM --------
                .task("reductionOneBlock", TornadoVMCompute::reductionOneBlock, context, intermediateReduceFirst, state.wrapX, localSizeRMS, config.rmsNormEps)
                .task("normalize1", TornadoVMCompute::reductionOneBlock2, context, state.wrapXb, state.wrapX, weights.rms_att_weightFlat, intermediateReduceFirst, state.positionAndLayer, dim)
                // -------- QKV MATMULS --------
                .task("qmatmul", TornadoVMCompute::matmul,  state.wrapQ, state.wrapXb, weights.wqFlat, dim, dim, state.positionAndLayer)
                .task("kmatmul", TornadoVMCompute::matmul,  state.wrapK, state.wrapXb, weights.wkFlat, dim, kvDim, state.positionAndLayer)
                .task("vmatmul", TornadoVMCompute::matmul,  state.wrapV, state.wrapXb, weights.wvFlat, dim, kvDim, state.positionAndLayer)

                // -------- ROPE ROTATION --------
                .task("rope", TornadoVMCompute::ropeRotation, context, state.positionAndLayer, state.wrapQ, state.wrapK, kvDim, headSize)

                // -------- COPY TO CACHES --------
                .task("copyToKeyCache", TornadoVMCompute::copyToCache, state.wrapKeyCache, state.wrapK, state.positionAndLayer)
                .task("copyToValueCache", TornadoVMCompute::copyToCache, state.wrapValueCache, state.wrapV, state.positionAndLayer)

                // -------- MULTI-HEAD ATTENTION --------
                .task("parallel-attention", TornadoVMCompute::processHeadsParallel,
                        state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                        config.numberOfHeads, config.headSize, kvDim, kvMul, config.contextLength,
                        state.positionAndLayer, state.wrapAtt)

                .task("matmul1", TornadoVMCompute::matmul, state.wrapXb2, state.wrapXb, weights.woFlat, dim, dim, state.positionAndLayer)
                .task("residual1", TornadoVMCompute::addInPlace, state.wrapX, state.wrapXb2)

                // -------- FFN --------
                .task("reductionOneBlockFFN", TornadoVMCompute::reductionOneBlock, context, intermediateReduceTwo, state.wrapX, localSizeRMS, config.rmsNormEps)
                .task("normalizeFFN", TornadoVMCompute::reductionOneBlock2, context, state.wrapXb, state.wrapX, weights.rms_ffn_weightFlat, intermediateReduceTwo, state.positionAndLayer, dim)
                .task("projcectOne", TornadoVMCompute::matmul,   state.wrapHb,state.wrapXb, weights.w1Flat, dim, config.hiddenDim, state.positionAndLayer)
                .task("projectionThree", TornadoVMCompute::matmul, state.wrapHb2,state.wrapXb, weights.w3Flat, dim, config.hiddenDim, state.positionAndLayer)
                .task("silu_elementwise_mul", TornadoVMCompute::siluElemWiseMulActivation, config.hiddenDim, state.wrapHb, state.wrapHb2)
                .task("projectionTwo", TornadoVMCompute::matmul,  state.wrapXb, state.wrapHb, weights.w2Flat, config.hiddenDim, dim, state.positionAndLayer)
                .task("residual2", TornadoVMCompute::addInPlace, state.wrapX, state.wrapXb)
                .persistOnDevice(
                        state.wrapX,
                        state.positionAndLayer, context
                );

        taskGraphs.add(unifiedLayer.snapshot());

        // ===== FIXED TASK GRAPH FOR RMS NORM AND LOGITS PROJECTION =====
        TaskGraph fixedFinalRmsGraph = new TaskGraph("rms_logits")
                // Consume from the layer task graph with explicit device objects
                .consumeFromDevice(unifiedLayer.getTaskGraphName(), state.wrapX, state.positionAndLayer, context)

                // Transfer all necessary buffers to device
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        state.wrapLogits,           // Output logits buffer
                        intermediateReduceThree,     // Intermediate reduction buffer
                        weights.rms_final_weight_as_floatArray,  // Final RMS norm weights
                        weights.wclsByteArray,       // Weight matrix for logits projection
                        debugBuffer)                 // Debug buffer

                // First RMS norm to normalize the input
                .task("reductionOneBlock",
                        TornadoVMCompute::reductionOneBlock,
                        context,
                        intermediateReduceThree,
                        state.wrapX,
                        localSizeRMS,
                        config.rmsNormEps)

                // Apply the normalization in-place on X
                .task("normalize",
                        TornadoVMCompute::reductionOneBlock2InNout,
                        context,
                        state.wrapX,
                        weights.rms_final_weight_as_floatArray,
                        intermediateReduceThree,
                        state.positionAndLayer,
                        dim)

                // Initialize logits buffer to avoid undefined values
                .task("initLogits",
                        (FloatArray output) -> {
                            for (@uk.ac.manchester.tornado.api.annotations.Parallel int i = 0; i < output.getSize(); i++) {
                                output.set(i, 0.0f);
                            }
                        },
                        state.wrapLogits)

                // Use the enhanced fixed matmul for Q8 weights with debugging
                .task("projection",
                        TornadoVMCompute::enhancedMatmulTornadoQ8,
                        context,
                        weights.wclsByteArray,
                        state.wrapX,
                        state.wrapLogits,
                        dim,
                        debugBuffer)

                // If debug mode is enabled, add a task to verify values
                .task("check",
                        TornadoVMCompute::checkLogitsValues,
                        state.wrapLogits,
                        debugBuffer)

                // Important: Explicitly transfer the result back to host after EVERY execution
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits, debugBuffer);

        taskGraphs.add(fixedFinalRmsGraph.snapshot());

        return new Tuple2<>(taskGraphs, setupGridSchedulers());
    }

    private int validateAndAdjustBufferSizes() {
        // Log dimensions
        System.out.println("Model dimensions:");
        System.out.println("dim = " + config.dim);
        System.out.println("headSize = " + config.headSize);
        System.out.println("numHeads = " + config.numberOfHeads);
        System.out.println("numKVHeads = " + config.numberOfKeyValueHeads);
        System.out.println("vocabularySize = " + config.vocabularySize);

        // Validate localSizeRMS
        int localSizeRMS = 256;
        if (config.dim % localSizeRMS != 0) {
            // Find a divisor close to original localSizeRMS
            for (int i = localSizeRMS; i > 0; i--) {
                if (config.dim % i == 0) {
                    localSizeRMS = i;
                    break;
                }
            }
        }

        // Return the expected reduction size
        return config.dim / localSizeRMS;
    }
}