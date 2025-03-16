package com.example.tornadovm;

import com.example.aux.Tuple2;
import com.example.core.model.tensor.FloatTensor;
import com.example.inference.engine.impl.Configuration;
import com.example.inference.engine.impl.Llama;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

public class TornadoVMLayerPlanner {


    private  final State state;
    private final Llama model;
    private final Configuration configuration;
    private final Weights weights;
    
    public TornadoVMLayerPlanner(State state, Llama model) {
        this.state = state;
        this.model = model;
        this.configuration = model.configuration();
        this.weights = model.weights();
    }


    public Tuple2<TornadoExecutionPlan, GridScheduler> createTornadoExecutionPlan() {

        TaskGraph taskGraph;
        KernelContext context = new KernelContext();
        boolean isQ4Type = weights.wcls.toString().contains("Q4");

        taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapXFloat)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.wclsByteArray)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, configuration.dim)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, configuration.vocabularySize)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);

        if (false) {
                taskGraph.task("t0", TornadoVMCompute::matmulTornadoQ4, context, weights.wclsByteArray, state.wrapXFloat, state.wrapLogits, configuration.dim);
//            taskGraph.task("t0", TornadoVMCompute::matrixVectorSimpleF15, state.wrapXFloat, state.wrapLogits, weights.halfFloat, configuration.dim, configuration.vocabularySize);
        } else {
            taskGraph.task("t0", TornadoVMCompute::matmulTornadoQ8, context, weights.wclsByteArray, state.wrapXFloat, state.wrapLogits, configuration.dim);
        }


        WorkerGrid worker = new WorkerGrid1D(configuration.vocabularySize);
        worker.setLocalWork(TornadoVMCompute.WORKGROUP,1,1);
        GridScheduler gridScheduler = new GridScheduler("s0.t0", worker);

        return new Tuple2<>(new TornadoExecutionPlan(taskGraph.snapshot()), gridScheduler);
    }

    /**
     * Creates a TornadoVM-based fused execution plan for processixng using a series of compute tasks
     * on the `TaskGraph`, optimizing tensor calculations, normalization, and matrix multiplication.
     * The generated execution plan leverages TornadoVM for GPU-accelerated tasks and adjusts based
     * on the data type of the model's weights.
     *
     * <p>This method configures a `TaskGraph` with data transfers and tasks for reduction,
     * summation, normalization, and matrix multiplication, preparing data transfer
     * from and to the host at each execution step. It dynamically adjusts tasks depending
     * on whether the model uses Q4 or Q8 quantized weights.</p>
     *
     * <p>Finally, this method sets up worker grids for each task to optimize performance
     * on the selected hardware, and associates them with the appropriate tasks in a `GridScheduler`.</p>
     *
     * @return A tuple containing a `TornadoExecutionPlan` and a `GridScheduler`:
     *         - `TornadoExecutionPlan`: Defines the fused task graph with all tasks and transfers set up.
     *         - `GridScheduler`: Manages grid workers for each task, optimizing parallel processing.
     */
    private Tuple2<TornadoExecutionPlan, GridScheduler> createTornadoExecutionPlanFused() {

        TaskGraph taskGraph;
        KernelContext context = new KernelContext();
        boolean isQ4Type = weights.wcls.toString().contains("Q4");

        final int size = configuration.dim;
        final int localSize = 256;

        FloatArray reduce = new FloatArray(size / localSize);

        taskGraph = new TaskGraph("fused")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapXFloat)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                        weights.wclsByteArray, weights.rms_final_weight_as_floatArray,
                        configuration.vocabularySize, configuration.dim, configuration.rmsNormEps)
                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.wrapXFloat, reduce) //
                .task("sum", TornadoVMCompute::finalSum, context, reduce,configuration.dim, configuration.rmsNormEps) //
                .task("ns", TornadoVMCompute::normalizeAndScale, context, state.wrapXFloat, weights.rms_final_weight_as_floatArray, reduce, configuration.dim) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits, state.wrapXFloat);

        if (isQ4Type) {
            taskGraph.task("mv", TornadoVMCompute::matmulTornadoQ4, context, weights.wclsByteArray, state.wrapXFloat, state.wrapLogits, configuration.dim);
        } else {
            taskGraph.task("mv", TornadoVMCompute::matmulTornadoQ8, context, weights.wclsByteArray, state.wrapXFloat, state.wrapLogits, configuration.dim);
        }

        WorkerGrid worker = new WorkerGrid1D(size);
        worker.setGlobalWork(size, 1, 1);
        worker.setLocalWork(localSize, 1, 1);

        WorkerGrid finalTokenWorker = new WorkerGrid1D(configuration.vocabularySize);
        finalTokenWorker.setGlobalWork(configuration.vocabularySize, 1, 1);
        finalTokenWorker.setLocalWork(TornadoVMCompute.WORKGROUP,1,1);

        GridScheduler gridScheduler = new GridScheduler("fused.reduce", worker);
        gridScheduler.setWorkerGrid("fused.sum", new WorkerGrid1D(1));
        gridScheduler.setWorkerGrid("fused.ns", worker);
        gridScheduler.setWorkerGrid("fused.mv", finalTokenWorker);

        return new Tuple2<>(new TornadoExecutionPlan(taskGraph.snapshot()), gridScheduler);
    }

    private Tuple2<TaskGraph, GridScheduler> firstFusedLayer() {
        TaskGraph taskGraph;
        KernelContext context = new KernelContext();
        boolean isQ4Type = weights.wcls.toString().contains("Q4");
        int dim = configuration.dim;
        int headSize = configuration.headSize;
        int kvDim = (configuration.dim * configuration.numberOfKeyValueHeads) / configuration.numberOfHeads;
        int kvMul = configuration.numberOfHeads / configuration.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        final int size = configuration.dim;
        final int localSize = 256;

        FloatArray reduce = new FloatArray(size / localSize);

        taskGraph = new TaskGraph("fused")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapXFloat)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        weights.wclsByteArray,
                        weights.rms_final_weight_as_floatArray,
                        configuration.vocabularySize,
                        configuration.dim,
                        configuration.rmsNormEps,
                        state.wrapQ,
                        state.wrapK,
                        state.wrapV,
                        state.wrapAtt,
                        state.wrapKeyCache,
                        state.wrapValueCache
                )
                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.wrapXFloat, reduce) //
                .task("sum", TornadoVMCompute::finalSum, context, reduce,configuration.dim, configuration.rmsNormEps) //
                .task("ns", TornadoVMCompute::normalizeAndScale, context, state.wrapXFloat, weights.rms_final_weight_as_floatArray, reduce, configuration.dim)
                .task("matmul1", TornadoVMCompute::matrixVectorSimple, state.wrapXb, state.wrapQ, weights.wqFlat, configuration.dim, configuration.dim, state.layer) // check if kernel is access the right dims
                .task("matmul2", TornadoVMCompute::matrixVectorSimple, state.wrapXb, state.wrapK, weights.wkFlat, kvDim, configuration.dim, state.layer) //
                .task("matmul3", TornadoVMCompute::matrixVectorSimple, state.wrapXb, state.wrapV, weights.wvFlat, kvDim, configuration.dim, state.layer) //
                .task("rope", TornadoVMCompute::ropeRotation, context, state.position, state.wrapQ, state.wrapK, kvDim,headSize)
                .persistOnDevice(state.wrapLogits, state.wrapXFloat, state.wrapQ, state.wrapK);

        WorkerGrid worker = new WorkerGrid1D(size);
        worker.setGlobalWork(size, 1, 1);
        worker.setLocalWork(localSize, 1, 1);

        WorkerGrid finalTokenWorker = new WorkerGrid1D(configuration.vocabularySize);
        finalTokenWorker.setGlobalWork(configuration.vocabularySize, 1, 1);
        finalTokenWorker.setLocalWork(TornadoVMCompute.WORKGROUP,1,1);

        GridScheduler gridScheduler = new GridScheduler("fused.reduce", worker);
        gridScheduler.setWorkerGrid("fused.sum", new WorkerGrid1D(1));
        gridScheduler.setWorkerGrid("fused.ns", worker);
        gridScheduler.setWorkerGrid("fused.mv", finalTokenWorker);
        gridScheduler.setWorkerGrid("matmul-1", worker);
        gridScheduler.setWorkerGrid("matmul-2", worker);
        gridScheduler.setWorkerGrid("matmul-3", worker);
        gridScheduler.setWorkerGrid("rope", worker);
        return new Tuple2<>(taskGraph, gridScheduler);
    }

    private Tuple2<TaskGraph, GridScheduler> multiHeadedAttentionLayer() {

    }

    private Tuple2<TornadoExecutionPlan, GridScheduler> createTornadoExecutionPlanPerLayer(int l) {
        TaskGraph taskGraph;
        KernelContext context = new KernelContext();
        final int localSize = 256;
        FloatArray reduce = new FloatArray(state.wrapXFloat.getSize() / localSize);

        taskGraph = new TaskGraph("ffn-layer")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, //
                        state.wrapXFloat, //
                        state.wrapHb, state.wrapHb2, //
                        state.wrapXb, state.wrapXb2 //
                ) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                        weights.woAsFloatArray[l], weights.w1AsFloatArray[l], //
                        weights.w2AFloatArray[l], weights.w3AFloatArray[l], //
                        configuration.vocabularySize, configuration.dim, //
                        configuration.rmsNormEps, configuration.hiddenDim, //
                        reduce
                ) //

                // First matmul and residual
                .task("matmul0", TornadoVMCompute::matrixVectorSimple, state.wrapXb, state.wrapXb2, weights.woAsFloatArray[l], configuration.dim, configuration.dim) //
                .task("addInPlace", TornadoVMCompute::addInPlace, state.wrapXb2, state.wrapXFloat) //

                // RMSNorm sequence
                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.wrapXFloat, reduce) //
                .task("sum", TornadoVMCompute::finalSum, context, reduce, configuration.dim, configuration.rmsNormEps) //
                .task("ns", TornadoVMCompute::normalizeAndScale2, context, state.wrapXb, state.wrapXFloat, weights.rms_ffn_weight_as_floatArray[l], reduce, configuration.dim) //

                // Parallel matmuls with separate output buffers
                .task("matmul1", TornadoVMCompute::matrixVectorSimple,  state.wrapXb, state.wrapHb,weights.w1AsFloatArray[l], configuration.hiddenDim, configuration.dim) //
                .task("matmul3", TornadoVMCompute::matrixVectorSimple, state.wrapXb, state.wrapHb2, weights.w3AFloatArray[l], configuration.hiddenDim, configuration.dim) //

                // SiLU and multiplication
                .task("mapInPlace", TornadoVMCompute::mapInPlace, state.wrapHb) //
                .task("multInPlace", TornadoVMCompute::multiplyInPlace, state.wrapHb, state.wrapHb2) //

                // Final matmul and residual
                .task("matmul2", TornadoVMCompute::matrixVectorSimple, state.wrapHb, state.wrapXb, weights.w2AFloatArray[l], configuration.dim, configuration.hiddenDim) //
                .task("addInPlace2", TornadoVMCompute::addInPlace, state.wrapXb, state.wrapXFloat) //

                // Buffer need to copy back
                .transferToHost(DataTransferMode.EVERY_EXECUTION,
                        state.wrapXFloat, //
                        state.wrapHb, state.wrapHb2, //
                        state.wrapXb, state.wrapXb2 //
                );

        WorkerGrid worker = new WorkerGrid1D(configuration.dim);
        worker.setGlobalWork(configuration.dim, 1, 1);
        worker.setLocalWork(localSize, 1, 1);


        GridScheduler gridScheduler = new GridScheduler("ffn-layer.addInPlace", worker);
        gridScheduler.setWorkerGrid("ffn-layer.reduce", worker);
        gridScheduler.setWorkerGrid("ffn-layer.sum", new WorkerGrid1D(1));
        gridScheduler.setWorkerGrid("ffn-layer.ns", worker);

        return new Tuple2<>(new TornadoExecutionPlan(taskGraph.snapshot()), gridScheduler);
    }

    public ArrayList<Tuple2<TornadoExecutionPlan,GridScheduler>> setupAndGetTornadoVMExecutionPlans() {
        ArrayList<Tuple2<TornadoExecutionPlan,GridScheduler>> tornadoVMPlans = new ArrayList<>();

        int numLayers = configuration.numberOfLayers;
        for (int i = 0; i < numLayers; i++) {
            tornadoVMPlans.add(createTornadoExecutionPlanPerLayer(i));
        }

        // plans.get(plans.size() - 1) -> size = numOfLayers +1;
        tornadoVMPlans.add(createTornadoExecutionPlanFused());
        return tornadoVMPlans;
    }


    /**
     * Set up and initialize all TornadoVM execution plans for LLM inference.
     * This method creates all task graphs, configures worker grids, and returns
     * a tuple containing the execution plan and grid scheduler that can be used in the forward method.
     */
    public static Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoExecutionPlans(
            Configuration config, Weights weights, State state) {

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

        FloatArray intermediateReduce = new FloatArray(dim / localSizeRMS);


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
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, state.x)
                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.wrapX, intermediateReduce, localSizeRMS)
                .task("sum", TornadoVMCompute::finalSum, context, intermediateReduce, dim, config.rmsNormEps)
                .task("normalize", TornadoVMCompute::normalizeAndScale, context,
                        state.wrapXb, state.wrapX, weights.rms_att_weightFlat, intermediateReduce, dim, config.rmsNormEps)
                .persistOnDevice(state.wrapX);

        gridScheduler.setWorkerGrid("rms-norm.reduce", dimWorker);
        gridScheduler.setWorkerGrid("rms-norm.sum", singleWorker);
        gridScheduler.setWorkerGrid("rms-norm.normalize", dimWorker);

        // Task Graph 1: QKV Matmuls
        TaskGraph qkvGraph = new TaskGraph("qkv")
                .consumeFromDevice(rmsNormGraph.getTaskGraphName(), state.xb)
                .task("q-matmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapQ,
                        weights.wqFlat, dim, dim, state.positionAndLayer)
                .task("k-matmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapK,
                        weights.wkFlat, dim, dim, state.positionAndLayer)
                .task("v-matmul", TornadoVMCompute::matrixVectorSimple, context, state.wrapXb, state.wrapV,
                        weights.wvFlat, dim, dim, state.positionAndLayer)
                .persistOnDevice(state.q, state.k, state.v);

        gridScheduler.setWorkerGrid("qkv.q-matmul", dimWorker);
        gridScheduler.setWorkerGrid("qkv.k-matmul", dimWorker);
        gridScheduler.setWorkerGrid("qkv.v-matmul", dimWorker);

        // Task Graph 2: RoPE
        TaskGraph ropeGraph = new TaskGraph("rope")
                .transferToDevice(DataTransferMode.UNDER_DEMAND, state.q, state.k)
                .task("rope", TornadoVMCompute::ropeRotation, context,
                        state.position, state.wrapQ,
                        state.wrapK, kvDim, headSize)
                .transferToHost(DataTransferMode.UNDER_DEMAND, state.q, state.k);

        gridScheduler.setWorkerGrid("rope.rope", ropeWorker);

        // Task Graph 3: Multi-head Attention
        // Important: The KV cache arrays are mapped to this graph from Graph 2 using device pointers
        TaskGraph attentionGraph = new TaskGraph("attention")
                // Attention memory is allocated on-device
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, state.att, state.maxValues,
                        state.expValues, state.sumValues)
                // KV cache arrays are mapped from previous graph (see mapOnDeviceMemoryRegion in forward method)
                .transferToDevice(DataTransferMode.UNDER_DEMAND, state.q)

                // Step 1: Calculate attention scores
                .task("scores", TornadoVMCompute::calculateAttentionScores, context,
                        state.position, config.contextLength, state.wrapQ, state.wrapKeyCache,
                        state.wrapAtt, kvDim, kvMul, headSize, 0)
                // Step 2: Find max for numerical stability
                .task("max", TornadoVMCompute::findMaxAttentionScores, context,
                        state.position, config.contextLength, state.wrapAtt, state.maxValues, localSizeHeads)
                // Step 3: Calculate exp and sum
                .task("expsum", TornadoVMCompute::calculateExpAndSum, context,
                        state.position, config.contextLength, state.wrapAtt, state.maxValues,
                        state.expValues, state.sumValues, localSizeHeads)
                // Step 4: Normalize with softmax
                .task("normalize", TornadoVMCompute::normalizeSoftmax, context,
                        state.position, config.contextLength, state.expValues,
                        state.sumValues, state.wrapAtt)
                // Step 5: Compute weighted sum
                .task("weighted-sum", TornadoVMCompute::computeWeightedSum, context,
                        state.position, config.contextLength, state.wrapAtt, state.wrapValueCache,
                        state.wrapXb, kvDim, kvMul, headSize, 0)
                .transferToHost(DataTransferMode.UNDER_DEMAND, state.xb);

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
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, state.hb, state.hb2, state.reduce)

                // Step 1: Matrix multiplication with attention output and residual
                .task("matmul1", TornadoVMCompute::matrixVectorMultiply,
                        context, state.xb, state.x, weights.woFlat, dim, dim)
                .task("residual1", TornadoVMCompute::addInPlace, context, state.x, state.xb)

                // Step 2: RMSNorm sequence
                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.xb, state.reduce, localSizeFFN)
                .task("sum", TornadoVMCompute::finalSum, context, state.reduce, dim, config.rmsNormEps)
                .task("ns", TornadoVMCompute::normalizeAndScale2,
                        context, state.x, state.xb, weights.rms_ffn_weightFlat, state.reduce, dim)

                // Step 3: Parallel projections with W1 and W3
                .task("projection1", TornadoVMCompute::matrixVectorMultiply,
                        context, state.x, state.hb, weights.w1Flat, dim, config.hiddenDim)
                .task("projection3", TornadoVMCompute::matrixVectorMultiply,
                        context, state.x, state.hb2, weights.w3Flat, dim, config.hiddenDim)

                // Step 4: SiLU activation and element-wise multiplication
                .task("silu", TornadoVMCompute::siluActivation, context, state.hb)
                .task("multiply", TornadoVMCompute::elementMultiply, context, state.hb2, state.hb)

                // Step 5: Final projection and residual
                .task("projection2", TornadoVMCompute::matrixVectorMultiply,
                        context, state.hb, state.x, weights.w2Flat, config.hiddenDim, dim)
                .task("residual2", TornadoVMCompute::addInPlace, context, state.x, state.xb)

                // Transfer result to host on-demand (will remain on device for next layer)
                .transferToHost(DataTransferMode.UNDER_DEMAND, state.xb);

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
        TaskGraph finalRmsNormGraph = new TaskGraph("final-rms")
                .transferToDevice(DataTransferMode.UNDER_DEMAND, state.x)
                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.wrapX, state.reduce, localSizeRMS)
                .task("sum", TornadoVMCompute::finalSum, context, state.reduce, dim, config.rmsNormEps)
                .task("normalize", TornadoVMCompute::normalizeAndScale, context,
                        state.wrapXFloat, state.wrapX, weights.rms_final_weight_as_floatArray,
                        state.reduce, dim, config.rmsNormEps)
                .transferToHost(DataTransferMode.UNDER_DEMAND, state.x);

        gridScheduler.setWorkerGrid("final-rms.reduce", dimWorker);
        gridScheduler.setWorkerGrid("final-rms.sum", singleWorker);
        gridScheduler.setWorkerGrid("final-rms.normalize", dimWorker);

        // Task Graph 6: Final Projection to Logits
        TaskGraph logitsGraph = new TaskGraph("logits")
                .transferToDevice(DataTransferMode.UNDER_DEMAND, state.x)
                .task("projection", TornadoVMCompute::matmulTornadoQ8, context,
                        weights.wclsByteArray, state.wrapXFloat, state.wrapLogits, dim)
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


        // Create execution plan with all graphs
//        TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(
//                immutableRMSGraph,         // Graph 0: RMSNorm
//                immutableQKVGraph,         // Graph 1: QKV Matmuls
//                immutableRopeGraph,        // Graph 2: RoPE
//                immutableAttentionGraph,   // Graph 3: Multi-head Attention
//                immutableFFNGraph,         // Graph 4: FFN
//                immutableFinalRMSGraph,    // Graph 5: Final RMSNorm
//                immutableLogitsGraph       // Graph 6: Final projection to logits
//        );

        // Return the execution plan and grid scheduler as a tuple
        return new Tuple2<>(taskGraphs, gridScheduler);
    }
}
