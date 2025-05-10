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

    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayered() {
        List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

        // Create kernel context
        KernelContext context = new KernelContext();

        state.temp.init(0.0f);
        state.tempFFN.init(0.0f);
        state.tempLogits.init(0.0f);

        // @formatter:off
        TaskGraph activationUpdate = new TaskGraph("activationUpdate")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                .task("updateX", TransformerComputeKernels::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX);
        taskGraphs.add(activationUpdate.snapshot());


        TaskGraph unifiedLayer = null;
        for (int i =0; i < config.numberOfLayers; i++) {
            unifiedLayer = new TaskGraph("layer_" + i);
            unifiedLayer.consumeFromDevice(state.wrapX) ;
            if (i==0) {
                unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        context,
                        state.wrapXb, state.wrapXb2,
                        state.wrapQ, state.wrapK, state.wrapV,
                        state.wrapKeyCache, state.wrapValueCache,
                        state.wrapAtt,
                        state.wrapHb);
            } else {
                unifiedLayer.consumeFromDevice(context,
                        state.wrapXb, state.wrapXb2,
                        state.wrapQ, state.wrapK, state.wrapV,
                        state.wrapKeyCache, state.wrapValueCache,
                        state.wrapAtt,
                        state.wrapHb);
            }
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    //Copy-in weights per layer for batched-layerd layout
                    weights.rms_att_weightLayered[i],
                    weights.wqLayered[i],
                    weights.wkLayered[i],
                    weights.wvLayered[i],
                    weights.woLayered[i],
                    weights.rms_ffn_weightLayered[i],
                    weights.w1Layered[i],
                    weights.w2Layered[i],
                    weights.w3Layered[i]
                );
            if ((i) == 0) {
                unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                                state.positionAndLayer,
                                state.temp,
                                state.tempFFN
                );
            } else {
                unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.positionAndLayer
//                        state.temp,
//                        state.tempFFN
                );
            };
            unifiedLayer.task("reductionsOneBlock" , TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.temp,
                        state.wrapX, config.dim, config.rmsNormEps, state.localSize)
                .task("mapContext", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                        state.wrapX, weights.rms_att_weightLayered[i], state.temp)
                .task("qmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                        state.wrapXb,  state.wrapQ, weights.wqLayered[i], config.dim, config.dim, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("kmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                        state.wrapXb,  state.wrapK, weights.wkLayered[i], config.dim, config.kvDim, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("vmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                        state.wrapXb,   state.wrapV, weights.wvLayered[i], config.dim, config.kvDim,  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("rope", TransformerComputeKernelsLayered::ropeRotation,context,
                        state.positionAndLayer, state.wrapQ, state.wrapK, config.kvDim,
                        config.headSize)
                .task("copyToCaches", TransformerComputeKernelsLayered::copyToCache,
                        state.wrapKeyCache, state.wrapK,  state.wrapValueCache, state.wrapV, state.positionAndLayer)
                .task("parallel-attention", TransformerComputeKernelsLayered::processHeadsParallel,
                        state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                        config.numberOfHeads, config.headSize, config.kvDim, config.kvMul, config.vocabularySize,
                        state.positionAndLayer, state.wrapAtt)
                .task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                        state.wrapXb,  state.wrapX, weights.woLayered[i], config.dim, config.dim,  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.tempFFN,
                        state.wrapX, config.dim, config.rmsNormEps, state.localSize)
                .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                        state.wrapX, weights.rms_ffn_weightLayered[i], state.tempFFN)
                .task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context,
                        state.wrapXb,   state.wrapHb, weights.w1Layered[i], weights.w3Layered[i], config.dim, config.hiddenDim, state.positionAndLayer, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("projectionTwo", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                        state.wrapHb, state.wrapX, weights.w2Layered[i], config.hiddenDim, config.dim,  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .persistOnDevice(
                        state.wrapX
                );
            taskGraphs.add(unifiedLayer.snapshot());
        }
        
        TaskGraph lastUnifiedLayer = unifiedLayer;

        TaskGraph logits = new TaskGraph("logits")
                .consumeFromDevice(lastUnifiedLayer.getTaskGraphName(),
                        state.wrapX
                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.tempLogits
                )
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        context,
                        state.wrapLogits,
                        weights.wclsByteArray,
                        weights.rms_final_weight_as_floatArray
                )
                .task("reductionsOneBlockLogits", TransformerComputeKernels::reductionOneBlockWithLayer, context, state.tempLogits,
                        state.wrapX, config.dim, config.rmsNormEps, state.localSize)
                .task("mapContextLogits", TransformerComputeKernels::reductionOneBlock2WithLogits, context, state.wrapX,
                        weights.rms_final_weight_as_floatArray, state.tempLogits)
                .task("projection", TransformerComputeKernels::matmulTornadoQ8Optimized, context,
                        weights.wclsByteArray, state.wrapX, state.wrapLogits, config.dim)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        taskGraphs.add(logits.snapshot());
        // @formatter:on

        return new Tuple2<>(taskGraphs, setupGridSchedulersLayered());
    }


    private GridScheduler setupGridSchedulersLayered() {
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

        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim);
        rmsNormWorker.setGlobalWork(config.dim, 1, 1);  // Set global work size to total dimension
        rmsNormWorker.setLocalWork(256, 1, 1);         // Set local work size to 256 (standard efficient size)


        // Map workers to tasks
        tornadoForwardScheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);
        for (int i = 0; i < config.numberOfLayers; i++) {
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qmatmul", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kmatmul", configKvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vmatmul", configKvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".matmul1", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".projectionTwo", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", configHiddenDimRowMajorWorker);


            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);

            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);

            tornadoForwardScheduler.addWorkerGrid("logits.reductionsOneBlockLogits", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("logits.mapContextLogits", rmsNormWorker);

            // Copy to caches - using System.setProperty to set
            System.setProperty("layer_" + i + ".copyToCaches.global.workgroup.size", "2048");
            System.setProperty("layer_" + i + ".copyToCaches.local.workgroup.size", "128");
            // Parallel attention - using System.setProperty
            System.setProperty("layer_" + i + ".parallel-attention.local.workgroup.size", "4");

        }

        // config .vocabularySize Worker for Normaml access MatVec
        WorkerGrid vocabWorker = new WorkerGrid1D(config.vocabularySize);
        vocabWorker.setGlobalWork(config.vocabularySize, 1, 1);
        vocabWorker.setLocalWork(16, 1, 1);

        tornadoForwardScheduler.addWorkerGrid("logits.projection", vocabWorker);

        return tornadoForwardScheduler;
    }

}
