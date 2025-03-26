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

public class TornadoVMLayerPlanner {

    private final State state;
    private final Configuration config;
    private final Weights weights;

    public TornadoVMLayerPlanner(State state, Llama model) {
        this.state = state;
        this.config = model.configuration();
        this.weights = model.weights();

    }

    //    public List<ImmutableTaskGraph> setupForwardTaskgraphs() {
    //        int dim = config.dim;
    //        int headSize = config.headSize;
    //        int numHeads = config.numberOfHeads;
    //        int numKVHeads = config.numberOfKeyValueHeads;
    //        int kvDim = (dim * numKVHeads) / numHeads;
    //        int kvMul = numHeads / numKVHeads;
    //
    //        List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();
    //
    //        // Define worker grid sizes
    //        int localSizeRMS = 256;
    //        int localSizeHeads = 64;
    //        int localSizeFFN = 256;
    //
    //        FloatArray intermediateReduceFirst = new FloatArray(dim / localSizeRMS);
    //        FloatArray intermediateReduceTwo = new FloatArray(dim / localSizeRMS);
    //        FloatArray intermediateReduceThree = new FloatArray(dim / localSizeRMS);
    //
    //        int seqLen = headSize;
    //        FloatArray attScores = new FloatArray(seqLen);
    //        FloatArray maxValues = new FloatArray(1);
    //        FloatArray expValues = new FloatArray(seqLen);
    //        FloatArray sumValues = new FloatArray(1);
    //
    //        //        validateArraySizes(intermediateReduceFirst, localSizeRMS);
    //        validateAndAdjustBufferSizes();
    //        // Create kernel context
    //        KernelContext context = new KernelContext();
    //
    //        // --- Create Task Graphs ---
    //        // @formatter:off
//
//        // Task Graph -1: Hack to force copy -> If it works, we can remove this
//        TaskGraph lookUpBufferX = new TaskGraph("lookUpBufferX")
//                .transferToDevice(DataTransferMode.FIRST_EXECUTION, state.wrapX)
//                .task("forceUpdateXperToken", TornadoVMCompute::emptyTaskToForceCopyIn, state.wrapX)
//                .persistOnDevice(state.wrapX);
//
//
//        // Task Graph 0: RMSNorm
//        TaskGraph rmsNormGraph = new TaskGraph("rmsnorm")
//                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionAndLayer)
//                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.rms_att_weightFlat, state.wrapX)
//                .consumeFromDevice(lookUpBufferX.getTaskGraphName(), state.wrapX)
//                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.wrapX, intermediateReduceFirst, localSizeRMS)
//                .task("sum", TornadoVMCompute::finalSum, context, intermediateReduceFirst, dim, config.rmsNormEps)
//                .task("normalize", TornadoVMCompute::normalizeAndScale, context,
//                        state.wrapXb, state.wrapX, weights.rms_att_weightFlat, intermediateReduceFirst, dim, state.positionAndLayer)
//                .persistOnDevice(state.wrapX, state.wrapXb, state.positionAndLayer);
//
//
//        // Task Graph 1: QKV Matmuls
//        TaskGraph qkvGraph = new TaskGraph("qkv")
//                .consumeFromDevice(rmsNormGraph.getTaskGraphName(), state.wrapX, state.wrapXb, state.positionAndLayer)
//                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
//                        weights.wqFlat, weights.wkFlat, weights.wvFlat)
//                .task("qmatmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapQ,
//                        weights.wqFlat, dim, dim, state.positionAndLayer)
//                .task("kmatmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapK,
//                        weights.wkFlat, dim, dim, state.positionAndLayer)
//                .task("vmatmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapV,
//                        weights.wvFlat, dim, dim, state.positionAndLayer)
//                .persistOnDevice(state.wrapQ, state.wrapK, state.wrapV, state.positionAndLayer);
//
//
//        // Task Graph 2: RoPE
//        TaskGraph ropeGraph = new TaskGraph("rotation")
//                .consumeFromDevice(qkvGraph.getTaskGraphName(), state.wrapQ, state.wrapK, state.positionAndLayer)
//                .task("rope", TornadoVMCompute::ropeRotation, context,
//                        state.positionAndLayer, state.wrapQ,
//                        state.wrapK, kvDim, headSize)
//                .persistOnDevice(state.wrapQ, state.wrapK, state.positionAndLayer);
//
//
//        // Task Graph 3: Multi-head Attention
//        // Important: The KV cache arrays are mapped to this graph from Graph 2 using device pointers
//        TaskGraph attentionGraph = new TaskGraph("attention")
//                // Attention memory is allocated on-device
////                .transferToDevice(DataTransferMode.FIRST_EXECUTION, state.att, state.maxValues,
////                        state.expValues, state.sumValues)
//                .consumeFromDevice(ropeGraph.getTaskGraphName(), state.wrapQ, state.positionAndLayer)
//                .transferToDevice(DataTransferMode.UNDER_DEMAND, state.wrapKeyCache, state.wrapValueCache)
//
//                // Step 1: Calculate attention scores
//                .task("scores", TornadoVMCompute::calculateAttentionScores, context,
//                        state.positionAndLayer, config.contextLength, state.wrapQ, state.wrapKeyCache,
//                        state.wrapAtt, kvDim, kvMul, headSize, 0, 0)
//                // Step 2: Find max for numerical stability
//                .task("max", TornadoVMCompute::findMaxAttentionScores, context,
//                        state.positionAndLayer, config.contextLength, state.wrapAtt, maxValues, localSizeHeads)
//                // Step 3: Calculate exp and sum
//                .task("expsum", TornadoVMCompute::calculateExpAndSum, context,
//                        state.positionAndLayer, config.contextLength, state.wrapAtt, maxValues,
//                        expValues, sumValues, localSizeHeads)
//                // Step 4: Normalize with softmax
//                .task("normalize", TornadoVMCompute::normalizeSoftmax, context,
//                        state.positionAndLayer, config.contextLength, expValues,
//                        sumValues, state.wrapAtt)
//                // Step 5: Compute weighted sum
//                .task("weighted-sum", TornadoVMCompute::computeWeightedSum, context,
//                        state.positionAndLayer, config.contextLength, state.wrapAtt, state.wrapValueCache,
//                        state.wrapXb, kvDim, kvMul, headSize, 0)
//                .persistOnDevice(state.wrapXb);
//
//
//        // Task Graph 4: FFN
//        TaskGraph ffnGraph = new TaskGraph("ffn")
//                .consumeFromDevice(attentionGraph.getTaskGraphName(), state.wrapXb, state.positionAndLayer)
//                // Static arrays are transferred once
//                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
//                        weights.woFlat, weights.rms_ffn_weightFlat,
//                        weights.w1Flat, weights.w2Flat, weights.w3Flat)
//
//                // Step 1: Matrix multiplication with attention output and residual
//                .task("matmul1", TornadoVMCompute::matrixVectorMultiply,
//                        context, state.wrapXb, state.wrapX, weights.woFlat, dim, dim, state.positionAndLayer)
//                .task("residual1", TornadoVMCompute::addInPlace, context, state.wrapX, state.wrapXb)
//
//                // Step 2: RMSNorm sequence
//                .task("reduceFFN", TornadoVMCompute::reduceSquareSums, context, state.wrapXb, intermediateReduceTwo, localSizeFFN)
//                .task("sum", TornadoVMCompute::finalSum, context, intermediateReduceTwo, dim, config.rmsNormEps)
//                .task("ns", TornadoVMCompute::normalizeAndScale,
//                        context, state.wrapX, state.wrapXb, weights.rms_ffn_weightFlat,
//                        intermediateReduceTwo, dim, state.positionAndLayer)
//
//                // Step 3: Parallel projections with W1 and W3
//                .task("projcectOne", TornadoVMCompute::matrixVectorMultiply,
//                        context, state.wrapX, state.wrapHb, weights.w1Flat, dim, config.hiddenDim, state.positionAndLayer)
//                .task("projectionThree", TornadoVMCompute::matrixVectorMultiply,
//                        context, state.wrapX, state.wrapHb2, weights.w3Flat, dim, config.hiddenDim, state.positionAndLayer)
//
//                // Step 4: SiLU activation and element-wise multiplication
//                .task("silu", TornadoVMCompute::siluActivation, context, state.wrapHb)
//                .task("multiply", TornadoVMCompute::elementMultiply, context, state.wrapHb2, state.wrapHb)
//
//                // Step 5: Final projection and residual
//                .task("projectionTwo", TornadoVMCompute::matrixVectorMultiply, context,
//                        state.wrapHb, state.wrapXb, weights.w2Flat, config.hiddenDim, dim, state.positionAndLayer)
//                .task("residual2", TornadoVMCompute::addInPlace, context, state.wrapX, state.wrapXb)
//
//                // Transfer result to host on-demand (will remain on device for next layer)
//                .persistOnDevice(state.wrapX);
//
//        // Task Graph 5: Final RMSNorm
//        TaskGraph finalRmsNormGraph = new TaskGraph("finalrms")
//                .consumeFromDevice(ffnGraph.getTaskGraphName(), state.wrapX, state.positionAndLayer)
//                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.rms_final_weight_as_floatArray)
//
//                .task("reduceRMS", TornadoVMCompute::reduceSquareSums, context, state.wrapX, intermediateReduceThree, localSizeRMS)
//                .task("sum", TornadoVMCompute::finalSum, context, intermediateReduceThree, dim, config.rmsNormEps)
//                .task("normalize", TornadoVMCompute::normalizeAndScale, context,
//                        state.wrapX, state.wrapX,
//                        weights.rms_final_weight_as_floatArray,
//                        intermediateReduceThree, dim, state.positionAndLayer)
//
//                .persistOnDevice(state.wrapX);
//
//
//        // Task Graph 6: Final Projection to Logits
//        TaskGraph logitsGraph = new TaskGraph("logits")
//                .consumeFromDevice(finalRmsNormGraph.getTaskGraphName(), state.wrapX, state.positionAndLayer)
//                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.wclsByteArray)
//                .task("projection", TornadoVMCompute::matmulTornadoQ8, context,
//                        weights.wclsByteArray,
//                        state.wrapX, state.wrapLogits,
//                        dim)
//                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.logits);
//
//        // @formatter:on
//
//        // Create immutable task graphs
//        ImmutableTaskGraph immutableLookUpX = lookUpBufferX.snapshot();
//        ImmutableTaskGraph immutableRMSGraph = rmsNormGraph.snapshot();
//        ImmutableTaskGraph immutableQKVGraph = qkvGraph.snapshot();
//        ImmutableTaskGraph immutableRopeGraph = ropeGraph.snapshot();
//        ImmutableTaskGraph immutableAttentionGraph = attentionGraph.snapshot();
//        ImmutableTaskGraph immutableFFNGraph = ffnGraph.snapshot();
//        ImmutableTaskGraph immutableFinalRMSGraph = finalRmsNormGraph.snapshot();
//        ImmutableTaskGraph immutableLogitsGraph = logitsGraph.snapshot();
//
//        // Add task graphs to the list
//        taskGraphs.add(0, immutableLookUpX);
//        taskGraphs.add(1, immutableRMSGraph);
//        taskGraphs.add(2, immutableQKVGraph);
//        taskGraphs.add(3, immutableRopeGraph);
//        taskGraphs.add(4, immutableAttentionGraph);
//        taskGraphs.add(5, immutableFFNGraph);
//        taskGraphs.add(6, immutableFinalRMSGraph);
//        taskGraphs.add(7, immutableLogitsGraph);
//
//        // Return the execution plan and grid scheduler as a tuple
//        return taskGraphs;
//    }
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

    // Scheduler 0: lookUpBufferX
    tornadoForwardScheduler.addWorkerGrid("lookUpBufferX.forceUpdateXperToken", singleWorker);

    // Scheduler 1: RMSNorm
    tornadoForwardScheduler.addWorkerGrid("rmsnorm.reduce", dimWorker);
    tornadoForwardScheduler.addWorkerGrid("rmsnorm.sum", singleWorker);
    tornadoForwardScheduler.addWorkerGrid("rmsnorm.normalize", dimWorker);

    // Scheduler 2: QKV
    tornadoForwardScheduler.addWorkerGrid("qkv.qmatmul", dimWorker);
    tornadoForwardScheduler.addWorkerGrid("qkv.kmatmul", dimWorker);
    tornadoForwardScheduler.addWorkerGrid("qkv.vmatmul", dimWorker);

    // Scheduler 3: RoPE
    tornadoForwardScheduler.addWorkerGrid("rotation.rope", ropeWorker);

    // Scheduler 4: Attention
    tornadoForwardScheduler.addWorkerGrid("attention.scores", headsWorker);
    tornadoForwardScheduler.addWorkerGrid("attention.max", headsWorker);
    tornadoForwardScheduler.addWorkerGrid("attention.expsum", headsWorker);
    tornadoForwardScheduler.addWorkerGrid("attention.normalize", headsWorker);
    tornadoForwardScheduler.addWorkerGrid("attention.weighted-sum", headsWorker);

    // Scheduler 5: FFN
    tornadoForwardScheduler.addWorkerGrid("ffn.matmul1", dimWorker);
    tornadoForwardScheduler.addWorkerGrid("ffn.residual1", dimWorker);
    tornadoForwardScheduler.addWorkerGrid("ffn.reduceFFN", dimWorker);
    tornadoForwardScheduler.addWorkerGrid("ffn.sum", singleWorker);
    tornadoForwardScheduler.addWorkerGrid("ffn.ns", dimWorker);
    tornadoForwardScheduler.addWorkerGrid("ffn.projcectOne", hiddenDimWorker);
    tornadoForwardScheduler.addWorkerGrid("ffn.projectionThree", hiddenDimWorker);
    tornadoForwardScheduler.addWorkerGrid("ffn.silu", hiddenDimWorker);
    tornadoForwardScheduler.addWorkerGrid("ffn.multiply", hiddenDimWorker);
    tornadoForwardScheduler.addWorkerGrid("ffn.projectionTwo", dimWorker);
    tornadoForwardScheduler.addWorkerGrid("ffn.residual2", dimWorker);

    // Scheduler 6: Final RMSNorm
    tornadoForwardScheduler.addWorkerGrid("finalrms.reduceRMS", dimWorker);
    tornadoForwardScheduler.addWorkerGrid("finalrms.sum", singleWorker);
    tornadoForwardScheduler.addWorkerGrid("finalrms.normalize", dimWorker);

    // Scheduler 7: Logits
    tornadoForwardScheduler.addWorkerGrid("logits.projection", vocabWorker);

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
        int localSizeFFN = 256;

        int reduceArraySize = validateAndAdjustBufferSizes();

        FloatArray intermediateReduceFirst = new FloatArray(reduceArraySize);
        FloatArray intermediateReduceTwo = new FloatArray(reduceArraySize);
        FloatArray intermediateReduceThree = new FloatArray(reduceArraySize);

        FloatArray maxValues = new FloatArray(1);
        FloatArray expValues = new FloatArray(headSize);
        FloatArray sumValues = new FloatArray(1);

        // Create kernel context
        KernelContext context = new KernelContext();

        // @formatter:off
        // ================ TASK GRAPH 0: BUFFER INITIALIZATION ================
        TaskGraph lookUpBufferX = new TaskGraph("lookUpBufferX")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                .task("forceUpdateXperToken", TornadoVMCompute::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX);
        taskGraphs.add(lookUpBufferX.snapshot());

        // ================ TASK GRAPH 1: RMS NORM ================
        // Fixed to properly consume from previous graph
        TaskGraph rmsNormGraph = new TaskGraph("rmsnorm")
                .consumeFromDevice(lookUpBufferX.getTaskGraphName(), state.wrapX)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.rms_att_weightFlat).transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionAndLayer)
                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.wrapX, intermediateReduceFirst, localSizeRMS)
                .task("sum", TornadoVMCompute::finalSum, intermediateReduceFirst, dim, config.rmsNormEps)
                .task("normalize", TornadoVMCompute::normalizeAndScale, context, state.wrapXb, state.wrapX, weights.rms_att_weightFlat, intermediateReduceFirst, dim, state.positionAndLayer)
                .persistOnDevice(state.wrapX, state.wrapXb, state.positionAndLayer);
        taskGraphs.add(rmsNormGraph.snapshot());

        // ================ TASK GRAPH 2: QKV MATMULS ================
        // Properly consumes from previous graph and persists needed buffers
        TaskGraph qkvGraph = new TaskGraph("qkv")
                .consumeFromDevice(rmsNormGraph.getTaskGraphName(), state.wrapXb, state.positionAndLayer, state.wrapX)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.wqFlat, weights.wkFlat, weights.wvFlat)
                .task("qmatmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapQ, weights.wqFlat, dim, dim, state.positionAndLayer)
                .task("kmatmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapK, weights.wkFlat, dim, dim, state.positionAndLayer)
                .task("vmatmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapV, weights.wvFlat, dim, dim, state.positionAndLayer)
                .task("forcePropagation", TornadoVMCompute::forcePropagationOneArray, state.wrapX)
                .persistOnDevice(state.wrapX, state.wrapXb, state.wrapQ, state.wrapK, state.wrapV, state.positionAndLayer);
        taskGraphs.add(qkvGraph.snapshot());

        // ================ TASK GRAPH 3: ROPE ROTATION ================
        TaskGraph ropeGraph = new TaskGraph("rotation")
                .consumeFromDevice(qkvGraph.getTaskGraphName(), state.wrapX, state.wrapXb, state.wrapQ, state.wrapK, state.wrapV, state.positionAndLayer)
                .task("rope", TornadoVMCompute::ropeRotation, context, state.positionAndLayer, state.wrapQ, state.wrapK, kvDim, headSize)
                .task("forcePropagation", TornadoVMCompute::forcePropagationThreeArrays, state.wrapX, state.wrapXb, state.wrapV)
                .persistOnDevice(state.wrapX, state.wrapXb, state.wrapQ, state.wrapK, state.wrapV, state.positionAndLayer);
        taskGraphs.add(ropeGraph.snapshot());

        // ================ TASK GRAPH 4: MULTI-HEAD ATTENTION ================
        // FIXED: We need to properly persist the buffers after the rope rotation
        // and before the memory mapping operation
        TaskGraph attentionGraph = new TaskGraph("attention")
                .consumeFromDevice(ropeGraph.getTaskGraphName(), state.wrapX, state.wrapXb, state.wrapQ, state.wrapK, state.wrapV, state.positionAndLayer)
                .transferToDevice(DataTransferMode.UNDER_DEMAND, state.wrapKeyCache, state.wrapValueCache)
                .task("scores", TornadoVMCompute::calculateAttentionScores, context, state.positionAndLayer, config.contextLength, state.wrapQ, state.wrapKeyCache, state.wrapAtt, kvDim, kvMul,
                        headSize, 0, localSizeRMS)
                .task("max", TornadoVMCompute::findMaxAttentionScoress, context, state.positionAndLayer, config.contextLength, state.wrapAtt, maxValues, localSizeHeads)
                .task("expsum", TornadoVMCompute::calculateExpAndSum, context, state.positionAndLayer, config.contextLength, state.wrapAtt, maxValues, expValues, sumValues, localSizeHeads)
                .task("normalize", TornadoVMCompute::normalizeSoftmax, context, state.positionAndLayer, config.contextLength, expValues, sumValues, state.wrapAtt)
                .task("weighted-sum", TornadoVMCompute::computeWeightedSum, context, state.positionAndLayer, config.contextLength, state.wrapAtt, state.wrapValueCache, state.wrapXb, kvDim, kvMul,
                        headSize, 0)
                .task("forcePropagationAttention", TornadoVMCompute::forcePropagationOneArray, state.wrapX)
                .persistOnDevice(state.wrapXb, state.positionAndLayer, state.wrapX);
        taskGraphs.add(attentionGraph.snapshot());

        // ================ TASK GRAPH 5: FFN ================
        TaskGraph ffnGraph = new TaskGraph("ffn")
                .consumeFromDevice(attentionGraph.getTaskGraphName(), state.wrapXb, state.positionAndLayer, state.wrapX)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.woFlat, weights.rms_ffn_weightFlat, weights.w1Flat, weights.w2Flat, weights.w3Flat)
                .task("matmul1", TornadoVMCompute::matrixVectorMultiply, context, state.wrapXb, state.wrapXb2, weights.woFlat, dim, dim, state.positionAndLayer)
                .task("residual1", TornadoVMCompute::addInPlace, context, state.wrapX, state.wrapXb2)
                .task("reduceFFN", TornadoVMCompute::reduceSquareSums, context, state.wrapX, intermediateReduceTwo, localSizeFFN)
                .task("sum", TornadoVMCompute::finalSum, intermediateReduceTwo, dim, config.rmsNormEps)
                .task("ns", TornadoVMCompute::normalizeAndScale, context, state.wrapXb, state.wrapX, weights.rms_ffn_weightFlat, intermediateReduceTwo, dim, state.positionAndLayer)
                .task("projcectOne", TornadoVMCompute::matrixVectorMultiply, context, state.wrapXb, state.wrapHb, weights.w1Flat, dim, config.hiddenDim, state.positionAndLayer)
                .task("projectionThree", TornadoVMCompute::matrixVectorMultiply, context, state.wrapXb, state.wrapHb2, weights.w3Flat, dim, config.hiddenDim, state.positionAndLayer)
                .task("silu", TornadoVMCompute::siluActivation, context, state.wrapHb)
                .task("multiply", TornadoVMCompute::elementMultiply, context, state.wrapHb2, state.wrapHb)
                .task("projectionTwo", TornadoVMCompute::matrixVectorMultiply, context, state.wrapHb, state.wrapXb, weights.w2Flat, config.hiddenDim, dim, state.positionAndLayer)
                .task("residual2", TornadoVMCompute::addInPlace, context, state.wrapX, state.wrapXb).persistOnDevice(state.wrapX, state.positionAndLayer, state.wrapXb);
        taskGraphs.add(ffnGraph.snapshot());

        // ================ TASK GRAPH 6: FINAL RMS NORM ================
        TaskGraph finalRmsNormGraph = new TaskGraph("finalrms")
                .consumeFromDevice(ffnGraph.getTaskGraphName(), state.wrapX, state.positionAndLayer)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.rms_final_weight_as_floatArray)
                .task("reduceRMS", TornadoVMCompute::reduceSquareSums, context, state.wrapX, intermediateReduceThree, localSizeRMS)
                .task("sum", TornadoVMCompute::finalSum, intermediateReduceThree, dim, config.rmsNormEps)
                .task("normalize", TornadoVMCompute::normalizeAndScaleInNout, context, state.wrapX, weights.rms_final_weight_as_floatArray, intermediateReduceThree, dim, state.positionAndLayer)
                .persistOnDevice(state.wrapX, state.positionAndLayer);
        taskGraphs.add(finalRmsNormGraph.snapshot());

        // ================ TASK GRAPH 7: FINAL PROJECTION TO LOGITS ================
        TaskGraph logitsGraph = new TaskGraph("logits")
                .consumeFromDevice(finalRmsNormGraph.getTaskGraphName(), state.wrapX, state.positionAndLayer)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.wclsByteArray)
                .task("projection", TornadoVMCompute::matmulTornadoQ8, context, weights.wclsByteArray, state.wrapX, state.wrapLogits, dim)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        taskGraphs.add(logitsGraph.snapshot());

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
            System.out.println("WARNING: dim (" + config.dim + ") is not divisible by localSizeRMS (" + localSizeRMS + ")");
            // Find a divisor close to original localSizeRMS
            for (int i = localSizeRMS; i > 0; i--) {
                if (config.dim % i == 0) {
                    localSizeRMS = i;
                    break;
                }
            }
            System.out.println("Adjusted localSizeRMS to " + localSizeRMS);
        }
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;

        // Check intermediate array sizes
        int expectedReduceSize = config.dim / localSizeRMS;
        System.out.println("Expected intermediate reduce size = " + expectedReduceSize);
        System.out.println("wrapX size = " + state.wrapX.getSize());
        // Add validation in validateAndAdjustBufferSizes()
        System.out.println("Key cache size: " + state.wrapKeyCache.getSize());
        System.out.println("Expected key cache size: " +
                (config.numberOfLayers * config.contextLength * kvDim));
        // Similar checks for value cache
        return expectedReduceSize;
    }
}
