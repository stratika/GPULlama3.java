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

    /**
     * Set up and initialize all TornadoVM execution plans for LLM inference. This method creates all task graphs, configures worker grids, and returns a tuple containing the execution plan and grid
     * scheduler that can be used in the forward method.
     */
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlan() {

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

        // Create kernel context
        KernelContext context = new KernelContext();

        // --- Worker Grids ---
        WorkerGrid dimWorker = new WorkerGrid1D(dim);
        dimWorker.setGlobalWork(dim, 1, 1);
        dimWorker.setLocalWork(localSizeRMS, 1, 1);

        WorkerGrid headsWorker = new WorkerGrid1D(numHeads * localSizeHeads);
        headsWorker.setGlobalWork(numHeads * localSizeHeads, 1, 1);
        headsWorker.setLocalWork(localSizeHeads, 1, 1);

        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        WorkerGrid hiddenDimWorker = new WorkerGrid1D(config.hiddenDim);
        hiddenDimWorker.setGlobalWork(config.hiddenDim, 1, 1);
        hiddenDimWorker.setLocalWork(localSizeFFN, 1, 1);

        WorkerGrid vocabWorker = new WorkerGrid1D(config.vocabularySize);
        vocabWorker.setGlobalWork(config.vocabularySize, 1, 1);
        vocabWorker.setLocalWork(256, 1, 1);

        WorkerGrid ropeWorker = new WorkerGrid1D(dim / 2);
        ropeWorker.setGlobalWork(dim / 2, 1, 1);
        ropeWorker.setLocalWork(localSizeRMS / 2, 1, 1);

        // --- Configure Grid Scheduler ---
        GridScheduler gridScheduler = new GridScheduler();

        // --- Create Task Graphs ---

        // @formatter:off
        // Task Graph 0: RMSNorm
        TaskGraph rmsNormGraph = new TaskGraph("rms-norm")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, state.wrapX, state.positionAndLayer)
                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.wrapX, intermediateReduceFirst, localSizeRMS)
                .task("sum", TornadoVMCompute::finalSum, context, intermediateReduceFirst, dim, config.rmsNormEps)
                .task("normalize", TornadoVMCompute::normalizeAndScale, context,
                        state.wrapXb, state.wrapX, weights.rms_att_weightFlat, intermediateReduceFirst, dim, state.positionAndLayer)
                .persistOnDevice(state.wrapX, state.positionAndLayer);

        gridScheduler.setWorkerGrid("rms-norm.reduce", dimWorker);
        gridScheduler.setWorkerGrid("rms-norm.sum", singleWorker);
        gridScheduler.setWorkerGrid("rms-norm.normalize", dimWorker);

        // Task Graph 1: QKV Matmuls
        TaskGraph qkvGraph = new TaskGraph("qkv")
                .consumeFromDevice(rmsNormGraph.getTaskGraphName(), state.wrapX, state.positionAndLayer)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        weights.wqFlat, weights.wkFlat, weights.wvFlat)
                .task("q-matmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapQ,
                        weights.wqFlat, dim, dim, state.positionAndLayer)
                .task("k-matmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapK,
                        weights.wkFlat, dim, dim, state.positionAndLayer)
                .task("v-matmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapV,
                        weights.wvFlat, dim, dim, state.positionAndLayer)
                .persistOnDevice(state.wrapQ, state.wrapK, state.wrapV, state.positionAndLayer);

        gridScheduler.setWorkerGrid("qkv.q-matmul", dimWorker);
        gridScheduler.setWorkerGrid("qkv.k-matmul", dimWorker);
        gridScheduler.setWorkerGrid("qkv.v-matmul", dimWorker);

        // Task Graph 2: RoPE
        TaskGraph ropeGraph = new TaskGraph("rope")
                .consumeFromDevice(qkvGraph.getTaskGraphName(), state.wrapQ, state.wrapK, state.positionAndLayer)
                .task("rope", TornadoVMCompute::ropeRotation, context,
                        state.positionAndLayer, state.wrapQ,
                        state.wrapK, kvDim, headSize)
                .persistOnDevice(state.wrapQ, state.wrapK, state.positionAndLayer);

        gridScheduler.setWorkerGrid("rope.rope", ropeWorker);

        // Task Graph 3: Multi-head Attention
        // Important: The KV cache arrays are mapped to this graph from Graph 2 using device pointers
        TaskGraph attentionGraph = new TaskGraph("attention")
                // Attention memory is allocated on-device
//                .transferToDevice(DataTransferMode.FIRST_EXECUTION, state.att, state.maxValues,
//                        state.expValues, state.sumValues)
                .consumeFromDevice(ropeGraph.getTaskGraphName(), state.wrapQ, state.positionAndLayer)

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

        gridScheduler.setWorkerGrid("attention.scores", headsWorker);
        gridScheduler.setWorkerGrid("attention.max", headsWorker);
        gridScheduler.setWorkerGrid("attention.expsum", headsWorker);
        gridScheduler.setWorkerGrid("attention.normalize", headsWorker);
        gridScheduler.setWorkerGrid("attention.weighted-sum", headsWorker);

        // Task Graph 4: FFN
        TaskGraph ffnGraph = new TaskGraph("ffn")
                // Input arrays are transferred on-demand (results from previous graph)
                .transferToDevice(DataTransferMode.UNDER_DEMAND, state.xb)
                // Static arrays are transferred once
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, state.hb, state.hb2)

                // Step 1: Matrix multiplication with attention output and residual
                .task("matmul1", TornadoVMCompute::matrixVectorMultiply,
                        context, state.wrapXb, state.wrapX, weights.woFlat, dim, dim, state.positionAndLayer)
                .task("residual1", TornadoVMCompute::addInPlace, context, state.wrapX, state.wrapXb)

                // Step 2: RMSNorm sequence
                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.wrapXb, intermediateReduceTwo, localSizeFFN)
                .task("sum", TornadoVMCompute::finalSum, context, intermediateReduceTwo, dim, config.rmsNormEps)
                .task("ns", TornadoVMCompute::normalizeAndScale,
                        context, state.wrapX, state.wrapXb, weights.rms_ffn_weightFlat, intermediateReduceTwo, dim, state.positionAndLayer)

                // Step 3: Parallel projections with W1 and W3
                .task("projection1", TornadoVMCompute::matrixVectorMultiply,
                        context, state.wrapX, state.wrapHb, weights.w1Flat, dim, config.hiddenDim, state.positionAndLayer)
                .task("projection3", TornadoVMCompute::matrixVectorMultiply,
                        context, state.wrapX, state.wrapHb2, weights.w3Flat, dim, config.hiddenDim, state.positionAndLayer)

                // Step 4: SiLU activation and element-wise multiplication
                .task("silu", TornadoVMCompute::siluActivation, context, state.wrapHb)
                .task("multiply", TornadoVMCompute::elementMultiply, context, state.wrapHb2, state.wrapHb)

                // Step 5: Final projection and residual
                .task("projection2", TornadoVMCompute::matrixVectorMultiply,
                        context, state.wrapHb, state.wrapX, weights.w2Flat, config.hiddenDim, dim, state.positionAndLayer)
                .task("residual2", TornadoVMCompute::addInPlace, context, state.wrapX, state.wrapHb)

                // Transfer result to host on-demand (will remain on device for next layer)
                .persistOnDevice(state.xb);

        // @formatter:on

        // Set FFN worker grids
        gridScheduler.setWorkerGrid("ffn.matmul1", dimWorker);
        gridScheduler.setWorkerGrid("ffn.residual1", dimWorker);
        gridScheduler.setWorkerGrid("ffn.reduce", dimWorker);
        gridScheduler.setWorkerGrid("ffn.sum", singleWorker);
        gridScheduler.setWorkerGrid("ffn.ns", dimWorker);
        gridScheduler.setWorkerGrid("ffn.projection1", hiddenDimWorker);
        gridScheduler.setWorkerGrid("ffn.projection3", hiddenDimWorker);
        gridScheduler.setWorkerGrid("ffn.silu", hiddenDimWorker);
        gridScheduler.setWorkerGrid("ffn.multiply", hiddenDimWorker);
        gridScheduler.setWorkerGrid("ffn.projection2", dimWorker);
        gridScheduler.setWorkerGrid("ffn.residual2", dimWorker);

        // Task Graph 5: Final RMSNorm
        TaskGraph finalRmsNormGraph = new TaskGraph("final-rms").transferToDevice(DataTransferMode.UNDER_DEMAND, state.x)
                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.wrapX, intermediateReduceThree, localSizeRMS)
                .task("sum", TornadoVMCompute::finalSum, context, intermediateReduceThree, dim, config.rmsNormEps)
                .task("normalize", TornadoVMCompute::normalizeAndScale, context, state.wrapXFloat, state.wrapX,
                        weights.rms_final_weight_as_floatArray, intermediateReduceThree, dim, state.positionAndLayer)
                .transferToHost(DataTransferMode.UNDER_DEMAND, state.x);

        gridScheduler.setWorkerGrid("final-rms.reduce", dimWorker);
        gridScheduler.setWorkerGrid("final-rms.sum", singleWorker);
        gridScheduler.setWorkerGrid("final-rms.normalize", dimWorker);

        // Task Graph 6: Final Projection to Logits
        TaskGraph logitsGraph = new TaskGraph("logits").transferToDevice(DataTransferMode.UNDER_DEMAND, state.x)
                .task("projection", TornadoVMCompute::matmulTornadoQ8, context, weights.wclsByteArray, state.wrapXFloat, state.wrapLogits, dim)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.logits);

        gridScheduler.setWorkerGrid("logits.projection", vocabWorker);

        // Create immutable task graphs
        ImmutableTaskGraph immutableRMSGraph = rmsNormGraph.snapshot();
        ImmutableTaskGraph immutableQKVGraph = qkvGraph.snapshot();
        ImmutableTaskGraph immutableRopeGraph = ropeGraph.snapshot();
        ImmutableTaskGraph immutableAttentionGraph = attentionGraph.snapshot();
        ImmutableTaskGraph immutableFFNGraph = ffnGraph.snapshot();
        ImmutableTaskGraph immutableFinalRMSGraph = finalRmsNormGraph.snapshot();
        ImmutableTaskGraph immutableLogitsGraph = logitsGraph.snapshot();

        taskGraphs.add(0, immutableRMSGraph);
        taskGraphs.add(1, immutableQKVGraph);
        taskGraphs.add(2, immutableRopeGraph);
        taskGraphs.add(3, immutableAttentionGraph);
        taskGraphs.add(4, immutableFFNGraph);
        taskGraphs.add(5, immutableFinalRMSGraph);
        taskGraphs.add(6, immutableLogitsGraph);


        // Return the execution plan and grid scheduler as a tuple
        return new Tuple2<>(taskGraphs, gridScheduler);
    }
}
