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
//    private static final int LOCAL_RMS_MM = Integer.getInteger("logits.projection", 16);

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
                .task("updateX", TornadoVMCompute::emptyTaskToForceCopyIn, state.wrapX)
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
//                        state.wrapHb2 //<- no need for hb2 in fused
                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.positionAndLayer,
                        state.temp,
                        state.tempFFN
//                        state.tempLogits
                        )
                .task("reductionsOneBlock", TornadoVMCompute::reductionOneBlockWithLayer, context, state.temp,
                        state.wrapX, config.dim, config.rmsNormEps, state.localSize)
                .task("mapContext", TornadoVMCompute::reductionOneBlock2WithLayer, context, state.wrapXb,
                        state.wrapX, weights.rms_att_weightFlat, state.temp, state.positionAndLayer, config.dim)
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
                        config.numberOfHeads, config.headSize, config.kvDim, config.kvMul, config.vocabularySize,
                        state.positionAndLayer, state.wrapAtt)
//                .task("matmul1", TornadoVMCompute::matmulUnroll4,
//                        state.wrapXb2, state.wrapXb, weights.woFlat, config.dim, config.dim, state.positionAndLayer)
//                .task("residual1", TornadoVMCompute::addInPlace, state.wrapX, state.wrapXb2)

                // final matmul to get the output of the attention fused with residual connection back into x
                .task("matmul1", TornadoVMCompute::matmulUnroll4WithResidual,
                        state.wrapX, state.wrapXb, weights.woFlat, config.dim, config.dim, state.positionAndLayer)

                .task("reductionsOneBlockFFN", TornadoVMCompute::reductionOneBlockWithLayer, context, state.tempFFN,
                        state.wrapX, config.dim, config.rmsNormEps, state.localSize)
                .task("mapContextFFN", TornadoVMCompute::reductionOneBlock2WithLayer, context, state.wrapXb,
                        state.wrapX, weights.rms_ffn_weightFlat, state.tempFFN, state.positionAndLayer, config.dim)
                .task("projectOne", TornadoVMCompute::matmulUnroll4,
                        state.wrapHb, state.wrapXb, weights.w1Flat, config.dim, config.hiddenDim, state.positionAndLayer)
//                .task("projectionThree", TornadoVMCompute::matmulUnroll4,
//                        state.wrapHb2, state.wrapXb, weights.w3Flat, config.dim, config.hiddenDim, state.positionAndLayer)
//                .task("silu_elementwise_mul", TornadoVMCompute::siluElemWiseMulActivation,
//                        config.hiddenDim, state.wrapHb, state.wrapHb2)
                .task("combinedProjectionAndActivation", TornadoVMCompute::combinedMatmulSiluActivation,context,
                        state.wrapHb, state.wrapXb, weights.w3Flat, config.dim, config.hiddenDim, state.positionAndLayer)
//                .task("projectionTwo", TornadoVMCompute::matmulUnroll4,
//                        state.wrapXb, state.wrapHb, weights.w2Flat, config.hiddenDim, config.dim, state.positionAndLayer)
//                .task("residual2", TornadoVMCompute::addInPlace, state.wrapX, state.wrapXb)

                // final matmul (down proj) to get the output of the ffn fused with residual connection back into x
//                .task("projectionTwo", TornadoVMCompute::projectionTwoOptimized, context,
//                state.wrapX, state.wrapHb, weights.w2Flat, config.hiddenDim, config.dim, state.positionAndLayer, 32)
                .task("projectionTwo", TornadoVMCompute::matmulUnroll4WithResidual, state.wrapX, state.wrapHb,
                        weights.w2Flat, config.hiddenDim, config.dim, state.positionAndLayer)

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
//                .task("reductionsOneBlockLogits", TornadoVMCompute::reductionOneBlockForLogits, context, state.tempLogits,
//                        state.wrapX, config.dim, config.rmsNormEps, state.localSize)
//                .task("mapContextLogits", TornadoVMCompute::applyNormForLogits, context,
//                        state.wrapX, weights.rms_att_weightFlat, config.dim, state.tempLogits)
//                        state.wrapX, weights.rms_att_weightFlat, config.dim, state.tempLogits)
                .task("rmsLogits", TornadoVMCompute::rmsnormInnOut,
                            state.wrapX, weights.rms_final_weight_as_floatArray, config.dim, config.rmsNormEps)
                .task("projection", TornadoVMCompute::matmulTornadoQ8Optimized, context, weights.wclsByteArray, state.wrapX, state.wrapLogits, config.dim)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
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
        vocabWorker.setLocalWork(16, 1, 1);

        WorkerGrid ropeWorker = new WorkerGrid1D(config.dim / 2);
        ropeWorker.setGlobalWork(config.dim / 2, 1, 1);
        ropeWorker.setLocalWork(128, 1, 1);

//        WorkerGrid projectionTwo = new WorkerGrid1D(config.dim * 32);
//        projectionTwo.setGlobalWork(config.dim * 32 , 1, 1);
//        projectionTwo.setLocalWork(32, 1, 1);
//



        tornadoForwardScheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);

        tornadoForwardScheduler.addWorkerGrid("layer.rope", ropeWorker);

        tornadoForwardScheduler.addWorkerGrid("logits.projection", vocabWorker);


        WorkerGrid projectionThree = new WorkerGrid1D(config.hiddenDim );
        projectionThree.setGlobalWork(config.hiddenDim , 1, 1);
        projectionThree.setLocalWork(128, 1, 1);

        tornadoForwardScheduler.addWorkerGrid("layer.combinedProjectionAndActivation", projectionThree);

//        tornadoForwardScheduler.addWorkerGrid("layer.projectionTwo", projectionTwo);

        // In your setupGridSchedulers method
        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim);
        rmsNormWorker.setGlobalWork(config.dim, 1, 1);  // Set global work size to total dimension
        rmsNormWorker.setLocalWork(256, 1, 1);         // Set local work size to 256 (standard efficient size)

        tornadoForwardScheduler.addWorkerGrid("layer.reductionsOneBlock", rmsNormWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.mapContext", rmsNormWorker);

        tornadoForwardScheduler.addWorkerGrid("layer.reductionsOneBlockFFN", rmsNormWorker);
        tornadoForwardScheduler.addWorkerGrid("layer.mapContextFFN", rmsNormWorker);


        tornadoForwardScheduler.addWorkerGrid("logits.reductionsOneBlockLogits", rmsNormWorker);
        tornadoForwardScheduler.addWorkerGrid("logits.mapContextLogits", rmsNormWorker);
        // Make sure to merge this scheduler with your tornadoForwardScheduler


        return tornadoForwardScheduler;
    }

}
