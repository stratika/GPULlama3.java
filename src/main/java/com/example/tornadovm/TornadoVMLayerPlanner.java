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

    public void validateArraySizes(FloatArray intermediateReduceFirst, int localSizeRMS) {
        // Log dimensions
        System.out.println("Model dimensions:");
        System.out.println("dim = " + config.dim);
        System.out.println("headSize = " + config.headSize);
        System.out.println("numHeads = " + config.numberOfHeads);
        System.out.println("numKVHeads = " + config.numberOfKeyValueHeads);

        // Check intermediate arrays
        System.out.println("intermediateReduceFirst size = " + intermediateReduceFirst.getSize());
        System.out.println("wrapX size = " + state.wrapX.getSize());

        // Validate reduce array size matches expected calculation pattern
        int expectedReduceSize = config.dim / localSizeRMS;
        if (intermediateReduceFirst.getSize() != expectedReduceSize) {
            System.out.println("WARNING: intermediateReduceFirst has incorrect size: " +
                    intermediateReduceFirst.getSize() + ", expected: " + expectedReduceSize);
        }
    }
    public Tuple2<List<ImmutableTaskGraph>, List<GridScheduler>> setupTornadoForwardPlan() {
        List<ImmutableTaskGraph> taskGraphs = setupForwardTaskgraphs();
        List<GridScheduler> schedulers = setupGridSchedulers(taskGraphs);
        return new Tuple2<>(taskGraphs, schedulers);
    }

    public List<ImmutableTaskGraph> setupForwardTaskgraphs() {
        int dim = config.dim;
        int headSize = config.headSize;
        int numHeads = config.numberOfHeads;
        int numKVHeads = config.numberOfKeyValueHeads;
        int kvDim = (dim * numKVHeads) / numHeads;
        int kvMul = numHeads / numKVHeads;

        List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

        // Define worker grid sizes
        int localSizeRMS = 256;
        int localSizeHeads = 64;
        int localSizeFFN = 256;

        FloatArray intermediateReduceFirst = new FloatArray(dim / localSizeRMS);
        FloatArray intermediateReduceTwo = new FloatArray(dim / localSizeRMS);
        FloatArray intermediateReduceThree = new FloatArray(dim / localSizeRMS);

        int seqLen = headSize;
        FloatArray attScores = new FloatArray(seqLen);
        FloatArray maxValues = new FloatArray(1);
        FloatArray expValues = new FloatArray(seqLen);
        FloatArray sumValues = new FloatArray(1);

        validateArraySizes(intermediateReduceFirst, localSizeRMS);
        // Create kernel context
        KernelContext context = new KernelContext();

        // --- Create Task Graphs ---
        // @formatter:off

        // Task Graph -1: Hack to force copy -> If it works, we can remove this
        TaskGraph lookUpBufferX = new TaskGraph("lookUpBufferX")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX, state.positionAndLayer)
                .task("forceUpdateXperToken", TornadoVMCompute::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX);

        // Task Graph 0: RMSNorm
        TaskGraph rmsNormGraph = new TaskGraph("rmsnorm")
                .consumeFromDevice(lookUpBufferX.getTaskGraphName(), state.wrapX)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionAndLayer)
                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.wrapX, intermediateReduceFirst, localSizeRMS)
                .task("sum", TornadoVMCompute::finalSum, context, intermediateReduceFirst, dim, config.rmsNormEps)
                .task("normalize", TornadoVMCompute::normalizeAndScale, context,
                        state.wrapXb, state.wrapX, weights.rms_att_weightFlat, intermediateReduceFirst, dim, state.positionAndLayer)
                .persistOnDevice(state.wrapX, state.positionAndLayer);


        // Task Graph 1: QKV Matmuls
        TaskGraph qkvGraph = new TaskGraph("qkv")
                .consumeFromDevice(rmsNormGraph.getTaskGraphName(), state.wrapX, state.positionAndLayer)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        weights.wqFlat, weights.wkFlat, weights.wvFlat)
                .task("qmatmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapQ,
                        weights.wqFlat, dim, dim, state.positionAndLayer)
                .task("kmatmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapK,
                        weights.wkFlat, dim, dim, state.positionAndLayer)
                .task("vmatmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapV,
                        weights.wvFlat, dim, dim, state.positionAndLayer)
                .persistOnDevice(state.wrapQ, state.wrapK, state.wrapV, state.positionAndLayer);


        // Task Graph 2: RoPE
        TaskGraph ropeGraph = new TaskGraph("rotation")
                .consumeFromDevice(qkvGraph.getTaskGraphName(), state.wrapQ, state.wrapK, state.positionAndLayer)
                .task("rope", TornadoVMCompute::ropeRotation, context,
                        state.positionAndLayer, state.wrapQ,
                        state.wrapK, kvDim, headSize)
                .persistOnDevice(state.wrapQ, state.wrapK, state.positionAndLayer);


        // Task Graph 3: Multi-head Attention
        // Important: The KV cache arrays are mapped to this graph from Graph 2 using device pointers
        TaskGraph attentionGraph = new TaskGraph("attention")
                // Attention memory is allocated on-device
//                .transferToDevice(DataTransferMode.FIRST_EXECUTION, state.att, state.maxValues,
//                        state.expValues, state.sumValues)
                .consumeFromDevice(ropeGraph.getTaskGraphName(), state.wrapQ, state.positionAndLayer)
                .transferToDevice(DataTransferMode.UNDER_DEMAND, state.wrapKeyCache, state.wrapValueCache)

                // Step 1: Calculate attention scores
                .task("scores", TornadoVMCompute::calculateAttentionScores, context,
                        state.positionAndLayer, config.contextLength, state.wrapQ, state.wrapKeyCache,
                        state.wrapAtt, kvDim, kvMul, headSize, 0)
                // Step 2: Find max for numerical stability
                .task("max", TornadoVMCompute::findMaxAttentionScores, context,
                        state.positionAndLayer, config.contextLength, state.wrapAtt, maxValues, localSizeHeads)
                // Step 3: Calculate exp and sum
                .task("expsum", TornadoVMCompute::calculateExpAndSum, context,
                        state.positionAndLayer, config.contextLength, state.wrapAtt, maxValues,
                        expValues, sumValues, localSizeHeads)
                // Step 4: Normalize with softmax
                .task("normalize", TornadoVMCompute::normalizeSoftmax, context,
                        state.positionAndLayer, config.contextLength, expValues,
                        sumValues, state.wrapAtt)
                // Step 5: Compute weighted sum
                .task("weighted-sum", TornadoVMCompute::computeWeightedSum, context,
                        state.positionAndLayer, config.contextLength, state.wrapAtt, state.wrapValueCache,
                        state.wrapXb, kvDim, kvMul, headSize, 0)
                .persistOnDevice(state.wrapXb);


        // Task Graph 4: FFN
        TaskGraph ffnGraph = new TaskGraph("ffn")
                .consumeFromDevice(attentionGraph.getTaskGraphName(), state.wrapXb, state.positionAndLayer)
                // Static arrays are transferred once
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        weights.woFlat, weights.rms_ffn_weightFlat,
                        weights.w1Flat, weights.w2Flat, weights.w3Flat)

                // Step 1: Matrix multiplication with attention output and residual
                .task("matmul1", TornadoVMCompute::matrixVectorMultiply,
                        context, state.wrapXb, state.wrapX, weights.woFlat, dim, dim, state.positionAndLayer)
                .task("residual1", TornadoVMCompute::addInPlace, context, state.wrapX, state.wrapXb)

                // Step 2: RMSNorm sequence
                .task("reduceFFN", TornadoVMCompute::reduceSquareSums, context, state.wrapXb, intermediateReduceTwo, localSizeFFN)
                .task("sum", TornadoVMCompute::finalSum, context, intermediateReduceTwo, dim, config.rmsNormEps)
                .task("ns", TornadoVMCompute::normalizeAndScale,
                        context, state.wrapX, state.wrapXb, weights.rms_ffn_weightFlat,
                        intermediateReduceTwo, dim, state.positionAndLayer)

                // Step 3: Parallel projections with W1 and W3
                .task("projcectOne", TornadoVMCompute::matrixVectorMultiply,
                        context, state.wrapX, state.wrapHb, weights.w1Flat, dim, config.hiddenDim, state.positionAndLayer)
                .task("projectionThree", TornadoVMCompute::matrixVectorMultiply,
                        context, state.wrapX, state.wrapHb2, weights.w3Flat, dim, config.hiddenDim, state.positionAndLayer)

                // Step 4: SiLU activation and element-wise multiplication
                .task("silu", TornadoVMCompute::siluActivation, context, state.wrapHb)
                .task("multiply", TornadoVMCompute::elementMultiply, context, state.wrapHb2, state.wrapHb)

                // Step 5: Final projection and residual
                .task("projectionTwo", TornadoVMCompute::matrixVectorMultiply, context,
                        state.wrapHb, state.wrapXb, weights.w2Flat, config.hiddenDim, dim, state.positionAndLayer)
                .task("residual2", TornadoVMCompute::addInPlace, context, state.wrapX, state.wrapXb)

                // Transfer result to host on-demand (will remain on device for next layer)
                .persistOnDevice(state.wrapX);

        // Task Graph 5: Final RMSNorm
        TaskGraph finalRmsNormGraph = new TaskGraph("finalrms")
                .consumeFromDevice(ffnGraph.getTaskGraphName(), state.wrapX, state.positionAndLayer)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.rms_final_weight_as_floatArray)

                .task("reduceRMS", TornadoVMCompute::reduceSquareSums, context, state.wrapX, intermediateReduceThree, localSizeRMS)
                .task("sum", TornadoVMCompute::finalSum, context, intermediateReduceThree, dim, config.rmsNormEps)
                .task("normalize", TornadoVMCompute::normalizeAndScale, context,
                        state.wrapX, state.wrapX,
                        weights.rms_final_weight_as_floatArray,
                        intermediateReduceThree, dim, state.positionAndLayer)

                .persistOnDevice(state.wrapX);


        // Task Graph 6: Final Projection to Logits
        TaskGraph logitsGraph = new TaskGraph("logits")
                .consumeFromDevice(finalRmsNormGraph.getTaskGraphName(), state.wrapX, state.positionAndLayer)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.wclsByteArray)
                .task("projection", TornadoVMCompute::matmulTornadoQ8, context,
                        weights.wclsByteArray,
                        state.wrapX, state.wrapLogits,
                        dim)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.logits);

        // @formatter:on

        // Create immutable task graphs
        ImmutableTaskGraph immutableLookUpX = lookUpBufferX.snapshot();
        ImmutableTaskGraph immutableRMSGraph = rmsNormGraph.snapshot();
        ImmutableTaskGraph immutableQKVGraph = qkvGraph.snapshot();
        ImmutableTaskGraph immutableRopeGraph = ropeGraph.snapshot();
        ImmutableTaskGraph immutableAttentionGraph = attentionGraph.snapshot();
        ImmutableTaskGraph immutableFFNGraph = ffnGraph.snapshot();
        ImmutableTaskGraph immutableFinalRMSGraph = finalRmsNormGraph.snapshot();
        ImmutableTaskGraph immutableLogitsGraph = logitsGraph.snapshot();

        // Add task graphs to the list
        taskGraphs.add(0, immutableLookUpX);
        taskGraphs.add(1, immutableRMSGraph);
        taskGraphs.add(2, immutableQKVGraph);
        taskGraphs.add(3, immutableRopeGraph);
        taskGraphs.add(4, immutableAttentionGraph);
        taskGraphs.add(5, immutableFFNGraph);
        taskGraphs.add(6, immutableFinalRMSGraph);
        taskGraphs.add(7, immutableLogitsGraph);

        // Return the execution plan and grid scheduler as a tuple
        return taskGraphs;
    }

    private List<GridScheduler> setupGridSchedulers(List<ImmutableTaskGraph> taskGraphs) {
        List<GridScheduler> schedulers = new ArrayList<>();

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
        GridScheduler lookupScheduler = new GridScheduler();
        lookupScheduler.setWorkerGrid("lookUpBufferX.forceUpdateXperToken", singleWorker);
        schedulers.add(lookupScheduler);

        // Scheduler 1: RMSNorm
        GridScheduler rmsNormScheduler = new GridScheduler();
        rmsNormScheduler.setWorkerGrid("rmsnorm.reduce", dimWorker);
        rmsNormScheduler.setWorkerGrid("rmsnorm.sum", singleWorker);
        rmsNormScheduler.setWorkerGrid("rmsnorm.normalize", dimWorker);
        schedulers.add(rmsNormScheduler);

        // Scheduler 2: QKV
        GridScheduler qkvScheduler = new GridScheduler();
        qkvScheduler.setWorkerGrid("qkv.qmatmul", dimWorker);
        qkvScheduler.setWorkerGrid("qkv.kmatmul", dimWorker);
        qkvScheduler.setWorkerGrid("qkv.vmatmul", dimWorker);
        schedulers.add(qkvScheduler);

        // Scheduler 3: RoPE
        GridScheduler ropeScheduler = new GridScheduler();
        ropeScheduler.setWorkerGrid("rotation.rope", ropeWorker);
        schedulers.add(ropeScheduler);

        // Scheduler 4: Attention
        GridScheduler attentionScheduler = new GridScheduler();
        attentionScheduler.setWorkerGrid("attention.scores", headsWorker);
        attentionScheduler.setWorkerGrid("attention.max", headsWorker);
        attentionScheduler.setWorkerGrid("attention.expsum", headsWorker);
        attentionScheduler.setWorkerGrid("attention.normalize", headsWorker);
        attentionScheduler.setWorkerGrid("attention.weighted-sum", headsWorker);
        schedulers.add(attentionScheduler);

        // Scheduler 5: FFN
        GridScheduler ffnScheduler = new GridScheduler();
        ffnScheduler.setWorkerGrid("ffn.matmul1", dimWorker);
        ffnScheduler.setWorkerGrid("ffn.residual1", dimWorker);
        ffnScheduler.setWorkerGrid("ffn.reduceFFN", dimWorker);
        ffnScheduler.setWorkerGrid("ffn.sum", singleWorker);
        ffnScheduler.setWorkerGrid("ffn.ns", dimWorker);
        ffnScheduler.setWorkerGrid("ffn.projcectOne", hiddenDimWorker);
        ffnScheduler.setWorkerGrid("ffn.projectionThree", hiddenDimWorker);
        ffnScheduler.setWorkerGrid("ffn.silu", hiddenDimWorker);
        ffnScheduler.setWorkerGrid("ffn.multiply", hiddenDimWorker);
        ffnScheduler.setWorkerGrid("ffn.projectionTwo", dimWorker);
        ffnScheduler.setWorkerGrid("ffn.residual2", dimWorker);
        schedulers.add(ffnScheduler);

        // Scheduler 6: Final RMSNorm
        GridScheduler finalRmsScheduler = new GridScheduler();
        finalRmsScheduler.setWorkerGrid("finalrms.reduceRMS", dimWorker);
        finalRmsScheduler.setWorkerGrid("finalrms.sum", singleWorker);
        finalRmsScheduler.setWorkerGrid("finalrms.normalize", dimWorker);
        schedulers.add(finalRmsScheduler);

        // Scheduler 7: Logits
        GridScheduler logitsScheduler = new GridScheduler();
        logitsScheduler.setWorkerGrid("logits.projection", vocabWorker);
        schedulers.add(logitsScheduler);

        return schedulers;
    }
}
