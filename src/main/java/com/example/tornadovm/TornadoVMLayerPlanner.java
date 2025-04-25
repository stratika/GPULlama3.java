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
import uk.ac.manchester.tornado.api.common.Access;
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

    private GridScheduler setupGridSchedulersX() {
        GridScheduler tornadoForwardScheduler = new GridScheduler();
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;

        // Create common worker grids that will be used across different schedulers
        WorkerGrid dimWorker = new WorkerGrid1D(config.dim);
        dimWorker.setGlobalWork(config.dim, 1, 1);
        dimWorker.setLocalWork(256, 1, 1);

        // Create common worker grids that will be used across different schedulers
        WorkerGrid kvdimWorker = new WorkerGrid1D(kvDim);
        dimWorker.setGlobalWork(kvDim, 1, 1);
        dimWorker.setLocalWork(128, 1, 1);

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

        tornadoForwardScheduler.addWorkerGrid("layer.rmsAtt", singleWorker);

        // Scheduler 3: RoPE Rotation
        tornadoForwardScheduler.addWorkerGrid("layer.rope", ropeWorker);

        // Scheduler 4: Copy to Caches
        tornadoForwardScheduler.addWorkerGrid("layer.copyToKeyCache", dimWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.copyToValueCache", dimWorker);

        // Scheduler 5: Multi-head Attention
        tornadoForwardScheduler.addWorkerGrid("layer.parallel-attention", headsWorker);

        // Scheduler 7: FFN Part 1 (Norm)
        tornadoForwardScheduler.addWorkerGrid("layer.rms", singleWorker);

        tornadoForwardScheduler.addWorkerGrid("logits.projection", vocabWorker);

        return tornadoForwardScheduler;
    }

    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlan() {
        List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

        // Create kernel context
        KernelContext context = new KernelContext();

        // @formatter:off

        // ================ TASK GRAPH 0: BUFFER INITIALIZATION ================
        TaskGraph updX = new TaskGraph("updX")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                .task("copyinX", TornadoVMCompute::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX);
        taskGraphs.add(updX.snapshot());

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
                        state.wrapHb, state.wrapHb2,
                        weights.freq_cis_realFlat, weights.freq_cis_imagFlat
                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.positionAndLayer
                )
                .task("rmsAtt", TornadoVMCompute::rmsnorm,
                        state.wrapXb, state.wrapX, weights.rms_att_weightFlat, state.positionAndLayer, config.dim, config.rmsNormEps)
                .task("qmatmul", TornadoVMCompute::matmul,
                        state.wrapQ, state.wrapXb, weights.wqFlat, config.dim, config.dim, state.positionAndLayer)
                .task("kmatmul", TornadoVMCompute::matmul,
                        state.wrapK, state.wrapXb, weights.wkFlat, config.dim, config.kvDim, state.positionAndLayer)
                .task("vmatmul", TornadoVMCompute::matmul,
                        state.wrapV, state.wrapXb, weights.wvFlat, config.dim, config.kvDim, state.positionAndLayer)
                .task("rope", TornadoVMCompute::ropeRotation,context,
                        state.positionAndLayer, state.wrapQ, state.wrapK, config.kvDim,
                        config.headSize)
                .task("copyToCaches", TornadoVMCompute::copyToCache,
                        state.wrapKeyCache, state.wrapK,  state.wrapValueCache, state.wrapV, state.positionAndLayer)
                .task("parallel-attention", TornadoVMCompute::processHeadsParallel,
                        state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                        config.numberOfHeads, config.headSize, config.kvDim, config.kvMul, config.contextLength,
                        state.positionAndLayer, state.wrapAtt)
                .task("matmul1", TornadoVMCompute::matmul,
                        state.wrapXb2, state.wrapXb, weights.woFlat, config.dim, config.dim, state.positionAndLayer)
                .task("residual1", TornadoVMCompute::addInPlace, state.wrapX, state.wrapXb2)
                .task("rms", TornadoVMCompute::rmsnorm,
                        state.wrapXb, state.wrapX, weights.rms_ffn_weightFlat, state.positionAndLayer, config.dim, config.rmsNormEps)
                .task("projectOne", TornadoVMCompute::matmul,
                        state.wrapHb, state.wrapXb, weights.w1Flat, config.dim, config.hiddenDim, state.positionAndLayer)
                .task("projectionThree", TornadoVMCompute::matmul,
                        state.wrapHb2, state.wrapXb, weights.w3Flat, config.dim, config.hiddenDim, state.positionAndLayer)
                .task("silu_elementwise_mul", TornadoVMCompute::siluElemWiseMulActivation,
                        config.hiddenDim, state.wrapHb, state.wrapHb2)
                .task("projectionTwo", TornadoVMCompute::matmul,
                        state.wrapXb, state.wrapHb, weights.w2Flat, config.hiddenDim, config.dim, state.positionAndLayer)
                .task("residual2", TornadoVMCompute::addInPlace, state.wrapX, state.wrapXb)
                .persistOnDevice(state.wrapX, context);
        taskGraphs.add(unifiedLayer.snapshot());


        TaskGraph logits = new TaskGraph("logits")
                .consumeFromDevice(unifiedLayer.getTaskGraphName(),
                        state.wrapX
                )
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        state.wrapLogits,
                        weights.wclsByteArray,
                        weights.rms_final_weight_as_floatArray
                )
                .task("rmsLogits", TornadoVMCompute::rmsnormInnOut,
                        state.wrapX, weights.rms_final_weight_as_floatArray, config.dim, config.rmsNormEps)
                .task("projection", TornadoVMCompute::matmulTornadoQ8, context, weights.wclsByteArray, state.wrapX, state.wrapLogits, config.dim)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits, state.wrapX);
        taskGraphs.add(logits.snapshot());
        // @formatter:on

        return new Tuple2<>(taskGraphs, setupGridSchedulers());
    }

    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanX() {
        List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();
        int dim = config.dim;
        int headSize = config.headSize;
        int numHeads = config.numberOfHeads;
        int numKVHeads = config.numberOfKeyValueHeads; // n_kv_heads
        int kvDim = (dim * numKVHeads) / numHeads;
        int kvMul = numHeads / numKVHeads;

        // Create kernel context
        KernelContext context = new KernelContext();

        // @formatter:off
        // ================ TASK GRAPH 0: BUFFER INITIALIZATION ================
        TaskGraph updX = new TaskGraph("updX")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                .task("copyinX", TornadoVMCompute::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapX);
        taskGraphs.add(updX.snapshot());

        // ================ TASK GRAPH 1: RMS NORM ================

        TaskGraph rmsNormGraph = new TaskGraph("rmsNorm")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        weights.rms_att_weightFlat,
                        state.wrapXb, state.wrapAtt,
                        state.wrapKeyCache, state.wrapValueCache,
                        state.wrapQ, state.wrapK, state.wrapV,
                        state.wrapXb2
                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionAndLayer, state.wrapX)
                .task("rmsAtt", TornadoVMCompute::rmsnorm,
                        state.wrapXb, state.wrapX, weights.rms_att_weightFlat, state.positionAndLayer, dim, config.rmsNormEps)
                .task("forceProp", TornadoVMCompute::forcePropagationTwoArrays, state.wrapKeyCache, state.wrapValueCache)
                .transferToHost(DataTransferMode.EVERY_EXECUTION,
                        state.wrapXb)
                .persistOnDevice(state.wrapXb, state.positionAndLayer, state.wrapAtt,
                        state.wrapKeyCache, state.wrapValueCache,
                        state.wrapQ, state.wrapK, state.wrapV,
                        state.wrapXb2);
        taskGraphs.add(rmsNormGraph.snapshot());

        // ================ TASK GRAPH 2: QKV MATMULS ================
        TaskGraph qkvMatmulGraph = new TaskGraph("qkvMatmul")
                .consumeFromDevice(rmsNormGraph.getTaskGraphName(),
                        state.wrapXb, state.wrapX)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        weights.wqFlat, weights.wkFlat, weights.wvFlat)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.wrapQ, state.wrapK, state.wrapV, state.positionAndLayer)
                .task("qmatmul", TornadoVMCompute::matmul,
                        state.wrapQ, state.wrapXb, weights.wqFlat, dim, dim, state.positionAndLayer)
                .task("kmatmul", TornadoVMCompute::matmul,
                        state.wrapK, state.wrapXb, weights.wkFlat, dim, kvDim, state.positionAndLayer)
                .task("vmatmul", TornadoVMCompute::matmul,
                        state.wrapV, state.wrapXb, weights.wvFlat, dim, kvDim, state.positionAndLayer)
                .persistOnDevice(state.wrapQ, state.wrapK, state.wrapV, state.wrapXb, state.wrapX,  weights.wqFlat, weights.wkFlat, weights.wvFlat)
                .transferToHost(DataTransferMode.EVERY_EXECUTION,
                        state.wrapQ, state.wrapK, state.wrapV);
        taskGraphs.add(qkvMatmulGraph.snapshot());

        // ================ TASK GRAPH 3: ROPE ROTATION ================
        TaskGraph ropeGraph = new TaskGraph("rp")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionAndLayer, context,   state.wrapQ, state.wrapK)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.freq_cis_realFlat, weights.freq_cis_imagFlat)
                .task("rope", TornadoVMCompute::ropeRotationSerial,
                         state.positionAndLayer, state.wrapQ, state.wrapK, config.numberOfHeads, config.numberOfKeyValueHeads, headSize, weights.freq_cis_realFlat, weights.freq_cis_imagFlat)
                .persistOnDevice(state.wrapQ, state.wrapK, state.positionAndLayer, state.wrapXb, state.wrapX,  state.wrapKeyCache, state.wrapValueCache)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapQ, state.wrapK, state.positionAndLayer);
        taskGraphs.add(ropeGraph.snapshot());

        // ================ TASK GRAPH 4: COPY TO CACHES ================
            TaskGraph copyToCachesGraph = new TaskGraph("copyToCaches")
                    .consumeFromDevice(state.wrapQ, state.wrapK, state.positionAndLayer, state.wrapXb, state.wrapX)
                    .task("copyToCaches", TornadoVMCompute::copyToCache,
                            state.wrapKeyCache, state.wrapK,  state.wrapValueCache, state.wrapV, state.positionAndLayer)
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


        // ================ TASK GRAPH 7: FFN PART 1 (NORM) ================
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


        TaskGraph finalRmsAndLogitsGraph = new TaskGraph("rms-final")
                .consumeFromDevice(
                        state.wrapX, state.positionAndLayer
                )
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        weights.rms_final_weight_as_floatArray)
                .task("rmsLogits", TornadoVMCompute::rmsnormInnOut,
                        state.wrapX, weights.rms_final_weight_as_floatArray,  dim, config.rmsNormEps)

                .transferToHost(DataTransferMode.EVERY_EXECUTION,  state.wrapX);
        taskGraphs.add(finalRmsAndLogitsGraph.snapshot());

        TaskGraph logits = new TaskGraph("logits")
                .consumeFromDevice(
                        state.wrapX, state.positionAndLayer
                )
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        state.wrapLogits, weights.wclsByteArray)
                .task("projection", TornadoVMCompute::matmulTornadoQ8, context, weights.wclsByteArray, state.wrapX, state.wrapLogits, dim)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        taskGraphs.add(logits.snapshot());
        // @formatter:on

        return new Tuple2<>(taskGraphs, setupGridSchedulersX());
    }

    private GridScheduler setupGridSchedulers() {
        GridScheduler tornadoForwardScheduler = new GridScheduler();
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;

        // Create common worker grids that will be used across different schedulers
        WorkerGrid dimWorker = new WorkerGrid1D(config.dim);
        dimWorker.setGlobalWork(config.dim, 1, 1);
        dimWorker.setLocalWork(256, 1, 1);

        // Create common worker grids that will be used across different schedulers
        WorkerGrid kvdimWorker = new WorkerGrid1D(kvDim);
        dimWorker.setGlobalWork(kvDim, 1, 1);
        dimWorker.setLocalWork(128, 1, 1);

        WorkerGrid headsWorker = new WorkerGrid1D(config.numberOfHeads * 64);
        headsWorker.setGlobalWork(config.numberOfHeads * 64, 1, 1);
        headsWorker.setLocalWork(64, 1, 1);

        WorkerGrid headsWorkerLiner = new WorkerGrid1D(config.numberOfHeads);
//        headsWorkerLiner.setGlobalWork(config.numberOfHeads, 1, 1);
//        headsWorker.setLocalWork(64, 1, 1);

        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        WorkerGrid hiddenDimWorker = new WorkerGrid1D(config.hiddenDim);
        hiddenDimWorker.setGlobalWork(config.hiddenDim, 1, 1);
        hiddenDimWorker.setLocalWork(256, 1, 1);

        WorkerGrid vocabWorker = new WorkerGrid1D(config.vocabularySize);
        vocabWorker.setGlobalWork(config.vocabularySize, 1, 1);
        vocabWorker.setLocalWork(192, 1, 1);

        WorkerGrid ropeWorker = new WorkerGrid1D(config.dim / 2);
        ropeWorker.setGlobalWork(config.dim / 2, 1, 1);
        ropeWorker.setLocalWork(128, 1, 1);

        // Scheduler 0: Buffer Initialization
        tornadoForwardScheduler.addWorkerGrid("updX.copyinX", singleWorker);

        // Scheduler 1: RMS Norm

        // Scheduler 3: RoPE Rotation
        tornadoForwardScheduler.addWorkerGrid("layer.rope", ropeWorker);

        // Scheduler 4: Copy to Caches

        // Scheduler 5: Multi-head Attention
//        tornadoForwardScheduler.addWorkerGrid("layer.parallel-attention", headsWorker);

        // Scheduler 7: FFN Part 1 (Norm)
        tornadoForwardScheduler.addWorkerGrid("logits.projection", vocabWorker);

        return tornadoForwardScheduler;
    }

}
