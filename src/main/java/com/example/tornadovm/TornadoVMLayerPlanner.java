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
//                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.wrapX, intermediateReduce, config.dim)
//                .task("finalSum", TornadoVMCompute::finalSum, context, intermediateReduceFirst, intermediateReduce, config.dim, config.rmsNormEps)
//                .task("normalize", TornadoVMCompute::normalizeAndScale, context, state.wrapXb, state.wrapX, weights.rms_att_weightFlat, intermediateReduceFirst, state.positionAndLayer, config.dim)

                .task("qmatmul", TornadoVMCompute::matmulUnroll4,
                        state.wrapQ, state.wrapXb, weights.wqFlat, config.dim, config.dim, state.positionAndLayer)
                .task("kmatmul", TornadoVMCompute::matmulUnroll4,
                        state.wrapK, state.wrapXb, weights.wkFlat, config.dim, config.kvDim, state.positionAndLayer)
                .task("vmatmul", TornadoVMCompute::matmulUnroll4,
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
                .task("matmul1", TornadoVMCompute::matmulUnroll4,
                        state.wrapXb2, state.wrapXb, weights.woFlat, config.dim, config.dim, state.positionAndLayer)
                .task("residual1", TornadoVMCompute::addInPlace, state.wrapX, state.wrapXb2)
                .task("rms", TornadoVMCompute::rmsnorm,
                        state.wrapXb, state.wrapX, weights.rms_ffn_weightFlat, state.positionAndLayer, config.dim, config.rmsNormEps)
                .task("projectOne", TornadoVMCompute::matmulUnroll4,
                        state.wrapHb, state.wrapXb, weights.w1Flat, config.dim, config.hiddenDim, state.positionAndLayer)
                .task("projectionThree", TornadoVMCompute::matmulUnroll4,
                        state.wrapHb2, state.wrapXb, weights.w3Flat, config.dim, config.hiddenDim, state.positionAndLayer)
                .task("silu_elementwise_mul", TornadoVMCompute::siluElemWiseMulActivation,
                        config.hiddenDim, state.wrapHb, state.wrapHb2)
                .task("projectionTwo", TornadoVMCompute::matmulUnroll4,
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




    private GridScheduler setupGridSchedulers() {
        GridScheduler tornadoForwardScheduler = new GridScheduler();

        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);


        WorkerGrid vocabWorker = new WorkerGrid1D(config.vocabularySize);
        vocabWorker.setGlobalWork(config.vocabularySize, 1, 1);
        vocabWorker.setLocalWork(64, 1, 1);

        WorkerGrid ropeWorker = new WorkerGrid1D(config.dim / 2);
        ropeWorker.setGlobalWork(config.dim / 2, 1, 1);
        ropeWorker.setLocalWork(128, 1, 1);


        tornadoForwardScheduler.addWorkerGrid("updX.copyinX", singleWorker);

        tornadoForwardScheduler.addWorkerGrid("layer.rope", ropeWorker);

        tornadoForwardScheduler.addWorkerGrid("logits.projection", vocabWorker);

        return tornadoForwardScheduler;
    }

}
