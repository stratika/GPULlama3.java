package com.example.tornadovm;

import com.example.aux.Tuple2;
import com.example.inference.engine.impl.Configuration;
import com.example.inference.engine.impl.Llama;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.ArrayList;
import java.util.List;

public class TornadoVMLayerPlannerDebug {
    private final State state;
    private final Configuration config;
    private final Weights weights;

    public TornadoVMLayerPlannerDebug(State state, Llama model) {
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

        WorkerGrid vocabWorker = new WorkerGrid1D(config.vocabularySize);
        vocabWorker.setGlobalWork(config.vocabularySize, 1, 1);
        vocabWorker.setLocalWork(256, 1, 1);

        WorkerGrid ropeWorker = new WorkerGrid1D(config.dim / 2);
        ropeWorker.setGlobalWork(config.dim / 2, 1, 1);
        ropeWorker.setLocalWork(128, 1, 1);

        // Create a scheduler for each task graph
        // Scheduler 0: updX
        tornadoForwardScheduler.addWorkerGrid("updX.copyinX", singleWorker);

        // Scheduler 1: RMSNorm

        tornadoForwardScheduler.addWorkerGrid("layer.reductionOneBlock", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.normalize1", dimWorker);

        // Scheduler 2: QKV

        // Scheduler 3: RoPE
        tornadoForwardScheduler.addWorkerGrid("layer.rope", ropeWorker);

        // Scheduler 4: Attention

        // Scheduler 5: FFN
        tornadoForwardScheduler.addWorkerGrid("layer.residual1", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.reductionOneBlockFFN", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.normalizeFNN", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.residual2", dimWorker);

//        // Scheduler 6: Final RMSNorm
//        tornadoForwardScheduler.addWorkerGrid("rms_logits.reductionOneBlock", dimWorker);
//        tornadoForwardScheduler.addWorkerGrid("rms_logits.normalize", dimWorker);
//
//        // Scheduler 7: Logits
//        tornadoForwardScheduler.addWorkerGrid("rms_logits.projection", vocabWorker);

        return tornadoForwardScheduler;
    }

    /**
     * Sets up the forward plan for TornadoVM execution with buffer sharing fixes. This implementation avoids array index out of bounds errors by: 1. Explicitly using the same references throughout
     * all task graphs 2. Avoiding consume operations on fields, using direct transfers instead
     */

    /**
     * Sets up the forward plan for TornadoVM execution with buffer sharing fixes. This implementation preserves the memory mapping approach for KV cache while fixing other buffer sharing issues.
     */
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlan() {
        List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

        // Use state arrays directly instead of storing local references

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


        // Create kernel context
        KernelContext context = new KernelContext();

        // @formatter:off
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
                        // Final RMS and logits
                        //                        intermediateReduceThree,
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
                .transferToHost(DataTransferMode.EVERY_EXECUTION,
                        // Main state variables that are modified
                        state.wrapX,           // Modified by residual1 and residual2
                        state.wrapXb,          // Modified by multiple tasks
                        state.wrapXb2,         // Modified by matmul1
                        state.wrapQ,           // Modified by qmatmul and rope
                        state.wrapK,           // Modified by kmatmul and rope
                        state.wrapV,           // Modified by vmatmul
                        state.wrapHb,          // Modified by projcectOne
                        state.wrapHb2,         // Modified by projectionThree and silu_elementwise_mul
                        // Intermediate results
                        intermediateReduceFirst,  // Modified by reductionOneBlock
                        intermediateReduceTwo,    // Modified by reductionOneBlockFFN
                        // Attention mechanism variables
                        state.wrapAtt,           // Modified by parallel-attention
                        maxValues,               // Used in attention calculation
                        expValues,               // Used in attention calculation
                        sumValues,               // Used in attention calculation
                        // Cache variables
                        state.wrapKeyCache,      // Modified by copyToKeyCache
                        state.wrapValueCache,    // Modified by copyToValueCache
                        // Position tracking
                        state.positionAndLayer  // Potentially modified during execution
                );


        // Execute PTX implementation

        taskGraphs.add(unifiedLayer.snapshot());
//        //        // ================ TASK GRAPH 6+7: FINAL RMS NORM AND LOGITS PROJECTION ================
//        TaskGraph finalRmsAndLogitsGraph = new TaskGraph("rms_logits")
//                .consumeFromDevice(unifiedLayer.getTaskGraphName(),
//                        state.wrapX, state.positionAndLayer, context
//                )
//                .transferToDevice(DataTransferMode.FIRST_EXECUTION, state.wrapLogits, intermediateReduceThree,  weights.rms_final_weight_as_floatArray,  weights.wclsByteArray)
//                .task("reductionOneBlock", TornadoVMCompute::reductionOneBlock, context, intermediateReduceThree, state.wrapX, localSizeRMS, config.rmsNormEps)
//                .task("normalize", TornadoVMCompute::reductionOneBlock2InNout, context, state.wrapX, weights.rms_final_weight_as_floatArray, intermediateReduceThree, state.positionAndLayer, dim)
//                .task("projection", TornadoVMCompute::matmulTornadoQ8, context, weights.wclsByteArray, state.wrapX, state.wrapLogits, dim)
//                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
//        taskGraphs.add(finalRmsAndLogitsGraph.snapshot());
        // @formatter:on

        return new Tuple2<>(taskGraphs, setupGridSchedulers());
    }

    private int validateAndAdjustBufferSizes() {
        // Log dimensions
        System.out.println("Model dimensions:");
        System.out.println("dim = " + config.dim);
        System.out.println("headSize = " + config.headSize);
        System.out.println("numHeads = " + config.numberOfHeads);
        System.out.println("numKVHeads = " + config.numberOfKeyValueHeads);

        // Validate localSizeRMS
        int localSizeRMS = 256;
        if (config.dim % localSizeRMS != 0) {
            //            System.out.println("WARNING: dim (" + config.dim + ") is not divisible by localSizeRMS (" + localSizeRMS + ")");
            // Find a divisor close to original localSizeRMS
            for (int i = localSizeRMS; i > 0; i--) {
                if (config.dim % i == 0) {
                    localSizeRMS = i;
                    break;
                }
            }
            //            System.out.println("Adjusted localSizeRMS to " + localSizeRMS);
        }
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;

        // Check intermediate array sizes
        int expectedReduceSize = config.dim / localSizeRMS;
        //        System.out.println("Expected intermediate reduce size = " + expectedReduceSize);
        //        System.out.println("wrapX size = " + state.wrapX.getSize());
        //        // Add validation in validateAndAdjustBufferSizes()
        //        System.out.println("Key cache size: " + state.wrapKeyCache.getSize());
        //        System.out.println("Expected key cache size: " + (config.numberOfLayers * config.contextLength * kvDim));
        // Similar checks for value cache
        return expectedReduceSize;
    }
}
