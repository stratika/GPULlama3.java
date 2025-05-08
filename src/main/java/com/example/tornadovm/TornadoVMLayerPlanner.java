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

import java.util.ArrayList;
import java.util.List;

public class TornadoVMLayerPlanner {
    private static final int LOCAL_WORK_GROUP_SIZE_ALLOC = 32;

    private final State state;
    private final Configuration config;
    private final Weights weights;

    public TornadoVMLayerPlanner(State state, Llama model) {
        this.state = state;
        this.config = model.configuration();
        this.weights = model.weights();
    }

    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlan() {
        List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

        // Create kernel context
        KernelContext context = new KernelContext();

        state.temp.init(0.0f);
        state.tempFFN.init(0.0f);
        state.tempLogits.init(0.0f);

        // @formatter:off
        // ================ TASK GRAPH 0: BUFFER INITIALIZATION ================
        TaskGraph activationUpdate = new TaskGraph("activationUpdate")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                .task("updateX", TransformerComputeKernels::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX);
        taskGraphs.add(activationUpdate.snapshot());

        // ================ TASK GRAPH 1: RMS NORM ================
        TaskGraph unifiedLayer = new TaskGraph("layer")
                .consumeFromDevice(state.wrapX)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        context,
                        weights.rms_att_weightFlat,
                        weights.wqFlat, weights.wkFlat, weights.wvFlat,
                        state.wrapXb, state.wrapXb2,
                        state.wrapQ, state.wrapK, state.wrapV,
                        state.wrapKeyCache, state.wrapValueCache,
                        state.wrapAtt,
                        weights.woFlat,
                        weights.rms_ffn_weightFlat,
                        weights.w1Flat, weights.w2Flat, weights.w3Flat,
                        state.wrapHb
                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.positionAndLayer,
                        state.temp,
                        state.tempFFN
                        )
                .task("reductionsOneBlock", TransformerComputeKernels::reductionOneBlockWithLayer, context, state.temp,
                        state.wrapX, config.dim, config.rmsNormEps, state.localSize)
                .task("mapContext", TransformerComputeKernels::reductionOneBlock2WithLayer, context, state.wrapXb,
                        state.wrapX, weights.rms_att_weightFlat, state.temp, state.positionAndLayer, config.dim)
                .task("qmatmul", TransformerComputeKernels::matrixVectorGeneric, context,
                        state.wrapXb,  state.wrapQ, weights.wqFlat, config.dim, config.dim, state.positionAndLayer, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("kmatmul", TransformerComputeKernels::matrixVectorGeneric, context,
                        state.wrapXb,  state.wrapK, weights.wkFlat, config.dim, config.kvDim, state.positionAndLayer,LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("vmatmul", TransformerComputeKernels::matrixVectorGeneric, context,
                       state.wrapXb,   state.wrapV, weights.wvFlat, config.dim, config.kvDim, state.positionAndLayer, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("rope", TransformerComputeKernels::ropeRotation,context,
                            state.positionAndLayer, state.wrapQ, state.wrapK, config.kvDim,
                        config.headSize)
                .task("copyToCaches", TransformerComputeKernels::copyToCache,
                        state.wrapKeyCache, state.wrapK,  state.wrapValueCache, state.wrapV, state.positionAndLayer)

                .task("parallel-attention", TransformerComputeKernels::processHeadsParallel,
                        state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                        config.numberOfHeads, config.headSize, config.kvDim, config.kvMul, config.vocabularySize,
                        state.positionAndLayer, state.wrapAtt)


                .task("matmul1", TransformerComputeKernels::matrixVectorGenericWithResidual, context,
                        state.wrapXb,  state.wrapX, weights.woFlat, config.dim, config.dim, state.positionAndLayer, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("reductionsOneBlockFFN", TransformerComputeKernels::reductionOneBlockWithLayer, context, state.tempFFN,
                        state.wrapX, config.dim, config.rmsNormEps, state.localSize)
                .task("mapContextFFN", TransformerComputeKernels::reductionOneBlock2WithLayer, context, state.wrapXb,
                        state.wrapX, weights.rms_ffn_weightFlat, state.tempFFN, state.positionAndLayer, config.dim)
                .task("fused_ffn_w1_w3", TransformerComputeKernels::fusedFeedForwardWithSiLUAndGLUActivation, context,
                        state.wrapXb,   state.wrapHb, weights.w1Flat, weights.w3Flat, config.dim, config.hiddenDim, state.positionAndLayer, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("projectionTwo", TransformerComputeKernels::matrixVectorGenericWithResidual, context,
                 state.wrapHb, state.wrapX, weights.w2Flat, config.hiddenDim, config.dim, state.positionAndLayer, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .persistOnDevice(state.wrapX, context);
        taskGraphs.add(unifiedLayer.snapshot());

        TaskGraph logits = new TaskGraph("logits")
                .consumeFromDevice(unifiedLayer.getTaskGraphName(),
                        state.wrapX
                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.tempLogits
                )
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        state.wrapLogits,
                        weights.wclsByteArray,
                        weights.rms_final_weight_as_floatArray
                )
                .task("reductionsOneBlockLogits", TransformerComputeKernels::reductionOneBlockWithLayer, context, state.tempLogits,
                        state.wrapX, config.dim, config.rmsNormEps, state.localSize)
                .task("mapContextLogits", TransformerComputeKernels::reductionOneBlock2WithLogits, context, state.wrapX,
                        weights.rms_final_weight_as_floatArray, state.tempLogits, state.positionAndLayer, config.dim)
                .task("projection", TransformerComputeKernels::matmulTornadoQ8Optimized, context,
                        weights.wclsByteArray, state.wrapX, state.wrapLogits, config.dim)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        taskGraphs.add(logits.snapshot());
        // @formatter:on

        return new Tuple2<>(taskGraphs, setupGridSchedulers());
    }

    private GridScheduler setupGridSchedulers() {
        GridScheduler tornadoForwardScheduler = new GridScheduler();

        // Single worker for tasks running with a single thread
        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        // config.dim / 2 Worker for RoPE
        WorkerGrid ropeWorker = new WorkerGrid1D(config.dim / 2);
        ropeWorker.setGlobalWork(config.dim / 2, 1, 1);
        ropeWorker.setLocalWork(128, 1, 1);

        // config.dim Worker for Row major access
        int configDimRowMajorGlobal = config.dim * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker = new WorkerGrid1D(configDimRowMajorGlobal);
        configDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // config.kvDim Worker for Row major access
        int configKvDimRowMajorGlobal = config.kvDim * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configKvDimRowMajorGlobalWorker = new WorkerGrid1D(configKvDimRowMajorGlobal);
        configKvDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // config.hiddenDim * 32 Worker for Row major access
        int configHiddenDimRowMajor = config.hiddenDim * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker = new WorkerGrid1D(configHiddenDimRowMajor);
        configHiddenDimRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // Map workers to tasks
        tornadoForwardScheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.qmatmul", configDimRowMajorGlobalWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.kmatmul", configKvDimRowMajorGlobalWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.vmatmul", configKvDimRowMajorGlobalWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.rope", ropeWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.matmul1", configDimRowMajorGlobalWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.projectionTwo", configDimRowMajorGlobalWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.fused_ffn_w1_w3", configHiddenDimRowMajorWorker);

        // config .vocabularySize Worker for Normaml access MatVec
        WorkerGrid vocabWorker = new WorkerGrid1D(config.vocabularySize);
        vocabWorker.setGlobalWork(config.vocabularySize, 1, 1);
        vocabWorker.setLocalWork(16, 1, 1);

//        WorkerGrid matrixQ8Worker = new WorkerGrid1D(config.vocabularySize * LOCAL_WORK_GROUP_SIZE_ALLOC);
//        matrixQ8Worker.setGlobalWork(config.vocabularySize * LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);
//        matrixQ8Worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);
//
//        tornadoForwardScheduler.addWorkerGrid("logits.projection", matrixQ8Worker);
        tornadoForwardScheduler.addWorkerGrid("logits.projection", vocabWorker);

        // Gridscheduler for all RMS normalizations
        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim);
        rmsNormWorker.setGlobalWork(config.dim, 1, 1);  // Set global work size to total dimension
        rmsNormWorker.setLocalWork(256, 1, 1);         // Set local work size to 256 (standard efficient size)

        tornadoForwardScheduler.addWorkerGrid("layer.reductionsOneBlock", rmsNormWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.mapContext", rmsNormWorker);

        tornadoForwardScheduler.addWorkerGrid("layer.reductionsOneBlockFFN", rmsNormWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.mapContextFFN", rmsNormWorker);

        tornadoForwardScheduler.addWorkerGrid("logits.reductionsOneBlockLogits", rmsNormWorker);
        tornadoForwardScheduler.addWorkerGrid("logits.mapContextLogits", rmsNormWorker);

//        // Add this to your GridScheduler configuration
//        WorkerGrid mhaWorker = new WorkerGrid1D(config.numberOfHeads);
//        mhaWorker.setGlobalWork(config.numberOfHeads, 1, 1);  // One workgroup per attention head
//        mhaWorker.setLocalWork(8, 1, 1);                   // 8 threads per workgroup
//
//        tornadoForwardScheduler.addWorkerGrid("parallel-attention", mhaWorker);

        return tornadoForwardScheduler;
    }

}
