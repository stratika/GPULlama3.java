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
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

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

    /**
     * Sets up the GridScheduler for the broken-down task graphs
     * @return The configured GridScheduler
     */
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

        // Scheduler 0: Buffer Initialization
        tornadoForwardScheduler.addWorkerGrid("updX.copyinX", singleWorker);

        // Scheduler 1: RMS Norm
//        tornadoForwardScheduler.addWorkerGrid("rmsNorm.reductionOneBlock", dimWorker);
//        tornadoForwardScheduler.addWorkerGrid("rmsNorm.normalize1", dimWorker);

        tornadoForwardScheduler.addWorkerGrid("layer.reduceSquareSums", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.finalizeReduction", singleWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.normalizeAndScale", dimWorker);
        // Scheduler 2: QKV Matmuls
        tornadoForwardScheduler.addWorkerGrid("layer.qmatmul", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.kmatmul", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.vmatmul", dimWorker);

        // Scheduler 3: RoPE Rotation
        tornadoForwardScheduler.addWorkerGrid("layer.rope", ropeWorker);

        // Scheduler 4: Copy to Caches
        tornadoForwardScheduler.addWorkerGrid("layer.copyToKeyCache", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.copyToValueCache", dimWorker);

        // Scheduler 5: Multi-head Attention
        tornadoForwardScheduler.addWorkerGrid("layer.parallel-attention", headsWorker);

        // Scheduler 6: Attention Output Processing
        tornadoForwardScheduler.addWorkerGrid("layer.matmul1", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.residual1", dimWorker);

        // Scheduler 7: FFN Part 1 (Norm)

        tornadoForwardScheduler.addWorkerGrid("layer.rms", singleWorker);
        //        tornadoForwardScheduler.addWorkerGrid("rmsNorm.reduceSquareSumsFFN", dimWorker);
//        tornadoForwardScheduler.addWorkerGrid("rmsNorm.finalizeReductionFFN", singleWorker);
//        tornadoForwardScheduler.addWorkerGrid("rmsNorm.normalizeAndScaleFFN", dimWorker);

        // Scheduler 8: FFN Part 2 (Projections)
        tornadoForwardScheduler.addWorkerGrid("layer.projectOne", hiddenDimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.projectionThree", hiddenDimWorker);

        // Scheduler 9: FFN Part 3 (Activation)
        tornadoForwardScheduler.addWorkerGrid("layer.silu_elementwise_mul", hiddenDimWorker);

        // Scheduler 10: FFN Part 4 (Final Projections)
        tornadoForwardScheduler.addWorkerGrid("layer.projectionTwo", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.residual2", dimWorker);

        return tornadoForwardScheduler;
    }

    private GridScheduler setupGridSchedulers2() {
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
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlan2() {
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
                .persistOnDevice(state.wrapX)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapX);
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

    /**
     * Sets up the forward plan for TornadoVM execution with buffer sharing fixes.
     * This implementation breaks down the unified task graph into multiple smaller ones
     * with explicit copy in/copy out operations for validation.
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
        int numGroups = (dim + localSizeRMS - 1) / localSizeRMS;
        int reduceArraySize = (config.dim + 256 - 1) / 256;

        reduceArraySize++;

        System.out.println("Reduce array size: " + reduceArraySize);

        FloatArray intermediateReduceFirst = new FloatArray(reduceArraySize);
        FloatArray intermediateReduceTwo = new FloatArray(reduceArraySize);
        FloatArray intermediateReduceThree = new FloatArray(reduceArraySize);

        intermediateReduceFirst.init(0.0f);
        intermediateReduceTwo.init(0.0f);
        intermediateReduceThree.init(0.0f);

        // Create kernel context
        KernelContext context = new KernelContext();

        // @formatter:off
        // ================ TASK GRAPH 0: BUFFER INITIALIZATION ================
        TaskGraph updX = new TaskGraph("updX")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, state.wrapX)
                .task("copyinX", TornadoVMCompute::emptyTaskToForceCopyIn, state.wrapX)
//                .persistOnDevice(state.wrapX)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapX);
        taskGraphs.add(updX.snapshot());

        // ================ TASK GRAPH 1: RMS NORM ================

        TaskGraph layer = new TaskGraph("layer")
//                .consumeFromDevice(state.wrapX)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        weights.rms_att_weightFlat,
                        weights.wqFlat, weights.wkFlat, weights.wvFlat,
                        intermediateReduceFirst,
                        state.wrapXb, state.wrapAtt,
                        state.wrapKeyCache, state.wrapValueCache,
                        state.wrapQ, state.wrapK, state.wrapV,
                        state.wrapXb2,
                        weights.woFlat,
                        weights.rms_ffn_weightFlat,
                        weights.w1Flat, weights.w3Flat,
                        weights.w2Flat, state.wrapHb, state.wrapHb2, context, state.wrapX
                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionAndLayer)
                .task("reduceSquareSums", TornadoVMCompute::reduceSquareSums, context,
                        intermediateReduceFirst, state.wrapX, localSizeRMS)
                .task("finalizeReduction", TornadoVMCompute::finalSum,
                        intermediateReduceFirst, dim, config.rmsNormEps)
                .task("normalizeAndScale", TornadoVMCompute::normalizeAndScale, context,
                        state.wrapXb, state.wrapX, weights.rms_att_weightFlat, intermediateReduceFirst, state.positionAndLayer, dim)
                .task("qmatmul", TornadoVMCompute::matmul,
                        state.wrapQ, state.wrapXb, weights.wqFlat, dim, dim, state.positionAndLayer)
                .task("kmatmul", TornadoVMCompute::matmul,
                        state.wrapK, state.wrapXb, weights.wkFlat, dim, kvDim, state.positionAndLayer)
                .task("vmatmul", TornadoVMCompute::matmul,
                        state.wrapV, state.wrapXb, weights.wvFlat, dim, kvDim, state.positionAndLayer)
                .task("rope", TornadoVMCompute::ropeRotation,
                        context, state.positionAndLayer, state.wrapQ, state.wrapK, kvDim, headSize)
                .task("copyToKeyCache", TornadoVMCompute::copyToCache,
                        state.wrapKeyCache, state.wrapK, state.positionAndLayer)
                .task("copyToValueCache", TornadoVMCompute::copyToCache,
                        state.wrapValueCache, state.wrapV, state.positionAndLayer)
                .task("parallel-attention", TornadoVMCompute::processHeadsParallel,
                state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                config.numberOfHeads, config.headSize, kvDim, kvMul, config.contextLength,
                state.positionAndLayer, state.wrapAtt)
                .task("matmul1", TornadoVMCompute::matmul,
                        state.wrapXb2, state.wrapXb, weights.woFlat, dim, dim, state.positionAndLayer)
                .task("residual1", TornadoVMCompute::addInPlace, state.wrapX, state.wrapXb2)
                .task("rms", TornadoVMCompute::rmsnorm,
                        state.wrapXb, state.wrapX, weights.rms_ffn_weightFlat, state.positionAndLayer, dim, config.rmsNormEps)
                .task("projectOne", TornadoVMCompute::matmul,
                        state.wrapHb, state.wrapXb, weights.w1Flat, dim, config.hiddenDim, state.positionAndLayer)
                .task("projectionThree", TornadoVMCompute::matmul,
                        state.wrapHb2, state.wrapXb, weights.w3Flat, dim, config.hiddenDim, state.positionAndLayer)
                .task("silu_elementwise_mul", TornadoVMCompute::siluElemWiseMulActivation,
                        config.hiddenDim, state.wrapHb, state.wrapHb2)
                .task("projectionTwo", TornadoVMCompute::matmul,
                        state.wrapXb, state.wrapHb, weights.w2Flat, config.hiddenDim, dim, state.positionAndLayer)
                .task("residual2", TornadoVMCompute::addInPlace, state.wrapX, state.wrapXb)
                .transferToHost(DataTransferMode.EVERY_EXECUTION,
                        state.wrapHb,
                        state.wrapHb2,
                        state.wrapX,
                        state.wrapXb);
//                .persistOnDevice(state.wrapXb, state.positionAndLayer,  weights.rms_att_weightFlat,
//                        weights.wqFlat, weights.wkFlat, weights.wvFlat,weights.woFlat,
//                        weights.rms_ffn_weightFlat,
//                        weights.w1Flat, weights.w3Flat,
//                        weights.w2Flat, context);
        taskGraphs.add(layer.snapshot());


        // @formatter:on

        return new Tuple2<>(taskGraphs, setupGridSchedulers());
    }


    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlan22() {
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
        int numGroups = (dim + localSizeRMS - 1) / localSizeRMS;
        int reduceArraySize = (config.dim + 256 - 1) / 256;

        reduceArraySize++;

        System.out.println("Reduce array size: " + reduceArraySize);

        FloatArray intermediateReduceFirst = new FloatArray(reduceArraySize);
        FloatArray intermediateReduceTwo = new FloatArray(reduceArraySize);
        FloatArray intermediateReduceThree = new FloatArray(reduceArraySize);

        intermediateReduceFirst.init(0.0f);
        intermediateReduceTwo.init(0.0f);
        intermediateReduceThree.init(0.0f);

        // Create kernel context
        KernelContext context = new KernelContext();

        // @formatter:off
        // ================ TASK GRAPH 0: BUFFER INITIALIZATION ================
        TaskGraph updX = new TaskGraph("updX")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, state.wrapX)
                .task("copyinX", TornadoVMCompute::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapX);
        taskGraphs.add(updX.snapshot());

        // ================ TASK GRAPH 1: RMS NORM ================

        TaskGraph rmsNormGraph = new TaskGraph("rmsNorm")
                .consumeFromDevice(state.wrapX)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        weights.rms_att_weightFlat,
                        intermediateReduceFirst
                        //                        state.wrapXb, state.wrapAtt,
                        //                        state.wrapKeyCache, state.wrapValueCache,
                        //                        state.wrapQ, state.wrapK, state.wrapV,
                        //                        state.wrapXb2

                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionAndLayer)
                .task("reduceSquareSums", TornadoVMCompute::reduceSquareSums, context,
                        intermediateReduceFirst, state.wrapX, localSizeRMS)
                .task("finalizeReduction", TornadoVMCompute::finalSum,
                        intermediateReduceFirst, dim, config.rmsNormEps)
                .task("normalizeAndScale", TornadoVMCompute::normalizeAndScale, context,
                        state.wrapXb, state.wrapX, weights.rms_att_weightFlat, intermediateReduceFirst, state.positionAndLayer, dim)
                .transferToHost(DataTransferMode.EVERY_EXECUTION,
                        state.wrapXb,
                        intermediateReduceFirst)
                .persistOnDevice(state.wrapXb, state.positionAndLayer);
        taskGraphs.add(rmsNormGraph.snapshot());

        // ================ TASK GRAPH 2: QKV MATMULS ================
        TaskGraph qkvMatmulGraph = new TaskGraph("qkvMatmul")
                .consumeFromDevice(state.wrapXb)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        weights.wqFlat, weights.wkFlat, weights.wvFlat)
                .task("qmatmul", TornadoVMCompute::matmul,
                        state.wrapQ, state.wrapXb, weights.wqFlat, dim, dim, state.positionAndLayer)
                .task("kmatmul", TornadoVMCompute::matmul,
                        state.wrapK, state.wrapXb, weights.wkFlat, dim, kvDim, state.positionAndLayer)
                .task("vmatmul", TornadoVMCompute::matmul,
                        state.wrapV, state.wrapXb, weights.wvFlat, dim, kvDim, state.positionAndLayer)
                .persistOnDevice(state.wrapQ, state.wrapK, state.wrapV, state.wrapXb, state.wrapX)
                .transferToHost(DataTransferMode.EVERY_EXECUTION,
                        state.wrapQ, state.wrapK, state.wrapV);
        taskGraphs.add(qkvMatmulGraph.snapshot());

        // ================ TASK GRAPH 3: ROPE ROTATION ================
        TaskGraph ropeGraph = new TaskGraph("rope")
                .consumeFromDevice(state.wrapQ, state.wrapK, state.positionAndLayer, state.wrapXb, state.wrapX)
                //                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionAndLayer)
                .task("rope", TornadoVMCompute::ropeRotation,
                        context, state.positionAndLayer, state.wrapQ, state.wrapK, kvDim, headSize)
                .persistOnDevice(state.wrapQ, state.wrapK, state.positionAndLayer, state.wrapXb, state.wrapX)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapQ, state.wrapK);
        taskGraphs.add(ropeGraph.snapshot());

        // ================ TASK GRAPH 4: COPY TO CACHES ================
        TaskGraph copyToCachesGraph = new TaskGraph("copyToCaches")
                .consumeFromDevice(state.wrapQ, state.wrapK, state.positionAndLayer, state.wrapXb, state.wrapX)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        state.wrapKeyCache, state.wrapValueCache)
                //                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionAndLayer)
                .task("copyToKeyCache", TornadoVMCompute::copyToCache,
                        state.wrapKeyCache, state.wrapK, state.positionAndLayer)
                .task("copyToValueCache", TornadoVMCompute::copyToCache,
                        state.wrapValueCache, state.wrapV, state.positionAndLayer)
                .persistOnDevice(state.wrapKeyCache, state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer, state.wrapXb, state.wrapX)
                .transferToHost(DataTransferMode.EVERY_EXECUTION,
                        state.wrapKeyCache, state.wrapValueCache);
        taskGraphs.add(copyToCachesGraph.snapshot());

        // ================ TASK GRAPH 5: MULTI-HEAD ATTENTION ================
        TaskGraph attentionGraph = new TaskGraph("attention")
                .consumeFromDevice(state.wrapKeyCache, state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer, state.wrapXb, state.wrapX)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        state.wrapAtt)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionAndLayer)
                .task("parallel-attention", TornadoVMCompute::processHeadsParallel,
                        state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                        config.numberOfHeads, config.headSize, kvDim, kvMul, config.contextLength,
                        state.positionAndLayer, state.wrapAtt)
                .persistOnDevice(state.wrapXb, state.wrapAtt, state.wrapKeyCache, state.wrapValueCache, state.wrapQ, state.wrapK,
                        state.positionAndLayer,  state.wrapX, context)
                .transferToHost(DataTransferMode.EVERY_EXECUTION,
                        state.wrapXb, state.wrapAtt);
        taskGraphs.add(attentionGraph.snapshot());

        // ================ TASK GRAPH 6: ATTENTION OUTPUT PROCESSING ================
        TaskGraph attOutGraph = new TaskGraph("attOutput")
                .consumeFromDevice(state.wrapXb, state.wrapAtt, state.wrapKeyCache, state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer,  state.wrapX)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        weights.woFlat, state.wrapXb2)
                .task("matmul1", TornadoVMCompute::matmul,
                        state.wrapXb2, state.wrapXb, weights.woFlat, dim, dim, state.positionAndLayer)
                .task("residual1", TornadoVMCompute::addInPlace, state.wrapX, state.wrapXb2)
                .persistOnDevice(state.wrapXb, state.wrapAtt, state.wrapKeyCache,
                        state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer,  state.wrapX, state.wrapXb2)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapX, state.wrapXb2);
        taskGraphs.add(attOutGraph.snapshot());


        // ADD A SERIAL VERSION OF THE ONE BELLOW JUST TO GO FURTHER
        // todo; llama.c

        // ================ TASK GRAPH 7: FFN PART 1 (NORM) ================
        //        TaskGraph ffnNormGraph = new TaskGraph("ffnNorm")
        //                .consumeFromDevice(state.wrapXb, state.wrapAtt, state.wrapKeyCache,
        //                        state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer,  state.wrapX, state.wrapXb2)
        //
        //                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
        //                        weights.rms_ffn_weightFlat, intermediateReduceTwo)
        //
        //                .task("reduceSquareSumsFFN", TornadoVMCompute::reduceSquareSums, context,
        //                        intermediateReduceTwo, state.wrapX, localSizeRMS)
        //                .task("finalizeReductionFFN", TornadoVMCompute::finalSum,
        //                        intermediateReduceTwo, dim, config.rmsNormEps)
        //                .task("normalizeAndScaleFFN", TornadoVMCompute::normalizeAndScale, context,
        //                        state.wrapXb, state.wrapX, weights.rms_ffn_weightFlat, intermediateReduceTwo, state.positionAndLayer, dim)
        //                .persistOnDevice(state.wrapXb,  state.wrapAtt, state.wrapKeyCache,
        //                        state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer,  state.wrapX, state.wrapXb2)
        //                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapXb, intermediateReduceTwo);
        //        taskGraphs.add(ffnNormGraph.snapshot());
        TaskGraph ffnNormGraph = new TaskGraph("rmsNormFFN")
                .consumeFromDevice(state.wrapXb, state.wrapAtt, state.wrapKeyCache,
                        state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer,  state.wrapX, state.wrapXb2)

                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        weights.rms_ffn_weightFlat)

                .task("rms", TornadoVMCompute::rmsnorm,
                        state.wrapXb, state.wrapX, weights.rms_ffn_weightFlat, state.positionAndLayer, dim, config.rmsNormEps)

                .persistOnDevice(state.wrapXb,  state.wrapAtt, state.wrapKeyCache,
                        state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer,  state.wrapX, state.wrapXb2)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapXb);
        taskGraphs.add(ffnNormGraph.snapshot());


        // ================ TASK GRAPH 8: FFN PART 2 (PROJECTIONS) ================
        TaskGraph ffnProjGraph = new TaskGraph("ffnProj")
                .consumeFromDevice(state.wrapXb,  state.wrapAtt, state.wrapKeyCache,
                        state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer,  state.wrapX, state.wrapXb2)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        weights.w1Flat, weights.w3Flat)
                .task("projectOne", TornadoVMCompute::matmul,
                        state.wrapHb, state.wrapXb, weights.w1Flat, dim, config.hiddenDim, state.positionAndLayer)
                .task("projectionThree", TornadoVMCompute::matmul,
                        state.wrapHb2, state.wrapXb, weights.w3Flat, dim, config.hiddenDim, state.positionAndLayer)
                .persistOnDevice(state.wrapHb, state.wrapHb2, state.wrapXb,  state.wrapAtt, state.wrapKeyCache,
                        state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer,  state.wrapX, state.wrapXb2)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapHb, state.wrapHb2);
        taskGraphs.add(ffnProjGraph.snapshot());

        // ================ TASK GRAPH 9: FFN PART 3 (ACTIVATION) ================
        TaskGraph ffnActivationGraph = new TaskGraph("ffnActivation")
                .consumeFromDevice( state.wrapXb,  state.wrapAtt, state.wrapKeyCache,
                        state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer,  state.wrapX, state.wrapXb2)
                .task("silu_elementwise_mul", TornadoVMCompute::siluElemWiseMulActivation,
                        config.hiddenDim, state.wrapHb, state.wrapHb2)
                .persistOnDevice(state.wrapHb, state.wrapHb2, state.wrapXb,  state.wrapAtt, state.wrapKeyCache,
                        state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer,  state.wrapX, state.wrapXb2)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapHb);
        taskGraphs.add(ffnActivationGraph.snapshot());

        // ================ TASK GRAPH 10: FFN PART 4 (FINAL PROJECTIONS) ================
        TaskGraph ffnFinalGraph = new TaskGraph("ffnFinal")
                .consumeFromDevice(state.wrapHb, state.wrapHb2, state.wrapXb,  state.wrapAtt, state.wrapKeyCache,
                        state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer,  state.wrapX, state.wrapXb2)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.w2Flat)
                .task("projectionTwo", TornadoVMCompute::matmul,
                        state.wrapXb, state.wrapHb, weights.w2Flat, config.hiddenDim, dim, state.positionAndLayer)
                .task("residual2", TornadoVMCompute::addInPlace, state.wrapX, state.wrapXb)
                .persistOnDevice(state.wrapX, state.wrapXb,state.wrapHb, state.wrapHb2, state.wrapXb,  state.wrapAtt, state.wrapKeyCache,
                        state.wrapValueCache, state.wrapQ, state.wrapK, state.positionAndLayer,  state.wrapX, state.wrapXb2 )
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapX, state.wrapXb);
        taskGraphs.add(ffnFinalGraph.snapshot());

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
