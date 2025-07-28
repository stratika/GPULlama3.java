package com.example.tornadovm;

import com.example.auxiliary.Tuple2;
import com.example.inference.state.Qwen3State;
import com.example.inference.state.State;
import com.example.inference.weights.tornado.Qwen3TornadoWeights;
import com.example.model.Model;
import com.example.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.ArrayList;
import java.util.List;

public class Qwen3TornadoVMLayerPlanner extends TornadoVMLayerPlanner<Qwen3State, Qwen3Configuration, Qwen3TornadoWeights> {

    int nHeadKv;
    int nEmbdHeadK;
    int nEmbdHeadV;
    int nEmbdVGqa;
    int nEmbdHead;
    int nEmbdGqa;
    int gqa;
    float sqrtHeadSize;

    public Qwen3TornadoVMLayerPlanner(Qwen3State state, Model model) {
        super(state, model);

        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHeadK = config.numberOfHeadsKey();
        this.nEmbdHeadV = config.numberOfHeadsValue(); // n_embd_head_v = n_embd / n_head; %s.attention.value_length
        this.nEmbdVGqa = nEmbdHeadV * nHeadKv; // n_embd_v_gqa = n_embd_head_v * n_head_kv
        this.nEmbdHead = nEmbdHeadV;
        this.nEmbdGqa = nEmbdVGqa;
        this.gqa = config.numberOfHeads() / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in multiquery
        this.sqrtHeadSize = (float) Math.sqrt(nEmbdHead);
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    state.positionHolder, state.temp, state.tempFFN,
                    state.tempQcur, state.tempKcur); //
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                    context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb);//,
                    // dbg buffers
                    //state.dbgQ, state.dbgKeyCache, state.dbgValueCache, state.dbgXb, state.dbgX); //
        } else {
            // Subsequent layers: Consume data already on device from previous layer
            unifiedLayer.consumeFromDevice(context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb, //
                    state.positionHolder //
                    //state.dbgQ, state.dbgKeyCache, state.dbgValueCache, state.dbgXb, state.dbgX
            );
        }
        return unifiedLayer;
    }

    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayered() {
        List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

        state.temp.init(0.0f);
        state.tempFFN.init(0.0f);
        state.tempLogits.init(0.0f);
        state.wrapLogits.init(0.0f);
        state.tempQcur.init(0.0f);
        state.tempKcur.init(0.0f);

//        state.dbgQ.init(0.0f);
//        state.dbgKeyCache.init(0.0f);
//        state.dbgValueCache.init(0.0f);

        // @formatter:off
        TaskGraph activationUpdate = new TaskGraph("activationUpdate")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                .task("updateX", TransformerComputeKernels::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX);
        taskGraphs.add(activationUpdate.snapshot());

        TaskGraph unifiedLayer = null;
        for (int layerIndex =0; layerIndex < config.numberOfLayers(); layerIndex++) {
            unifiedLayer = new TaskGraph("layer_" + layerIndex);
            unifiedLayer.consumeFromDevice(state.wrapX);
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    //Copy-in weights per layer for batched-layered layout
                    weights.rms_att_weightLayered[layerIndex],
                    weights.wqLayered[layerIndex],
                    weights.wkLayered[layerIndex],
                    weights.wvLayered[layerIndex],
                    weights.woLayered[layerIndex],
                    //rms_att_KNormLayered
                    weights.rms_att_KNormLayered[layerIndex],
                    //rms_att_QNormLayered
                    weights.rms_att_QNormLayered[layerIndex],
                    weights.rms_ffn_weightLayered[layerIndex],
                    weights.w1Layered[layerIndex],
                    weights.w2Layered[layerIndex],
                    weights.w3Layered[layerIndex]
            );
            unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
//            unifiedLayer.task("dbg_copy_out_x",
//                    Qwen3Kernels::dbgCopy,
//                    state.wrapX,
//                    state.dbgX,
//                    state.positionHolder,
//                    layerIndex);
            unifiedLayer.task("reductionsOneBlock",
                                    TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                                    context,
                                    state.temp,
                                    state.wrapX, // in
                                    config.dim(),
                                    config.rmsNormEps(),
                                    state.localSize)
                            //.task("reductionFinalNormalization" , TransformerComputeKernelsLayered::reductionFinalNormalization, context,
                                    //state.temp, config.dim(), config.rmsNormEps())
                            .task("mapContext",
                                    TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                                    context,
                                    state.wrapXb, // out
                                    state.wrapX,
                                    weights.rms_att_weightLayered[layerIndex],
                                    state.temp);

            //unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapXb);

//            // dbg copy out
//            unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.temp);
//            unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapXb);

            int qDim0 = nEmbdHeadK * config.numberOfHeads();
            int kvDim0 = nEmbdGqa;
            int qkvDim1 = config.dim();
            //qkvMatmuls = new TaskGraph("qkvMatmuls_layer_" + layerIndex);
            unifiedLayer.task("qmatmul",
                            TransformerComputeKernelsLayered::matrixVectorGeneric,
                            context,
                            state.wrapXb,
                            state.wrapQ,                    // output
                            weights.wqLayered[layerIndex],
                            qkvDim1,
                            qDim0,
                            LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("kmatmul",
                            TransformerComputeKernelsLayered::matrixVectorGeneric,
                            context,
                            state.wrapXb,
                            state.wrapK,        // output
                            weights.wkLayered[layerIndex],
                            qkvDim1,
                            kvDim0,
                            LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("vmatmul",
                            TransformerComputeKernelsLayered::matrixVectorGeneric,
                            context,
                            state.wrapXb,
                            state.wrapV,        // output
                            weights.wvLayered[layerIndex],
                            qkvDim1,
                            kvDim0,
                            LOCAL_WORK_GROUP_SIZE_ALLOC);

//            unifiedLayer.task("dbg_copy_out_wrapQ",
//                    Qwen3Kernels::dbgCopy,
//                    state.wrapQ,
//                    state.dbgQ,
//                    state.positionHolder,
//                    layerIndex);

            // dbg copy out
//            unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapQ);
//            unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapK);
//            unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapV);

            // Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            for (int i = 0; i < config.numberOfHeads(); i++) {
                //rmsnorm(state.q, state.q, weights.attnQNorm[curLayer], i * nEmbdHead, nEmbdHead, config.rmsNormEps());
                int offset = i * nEmbdHead;
                unifiedLayer.task("reductionsOneBlock" + "_Qcur_" + i ,
                                Qwen3Kernels::reductionOneBlockWithLayerWithOffset,
                                context,
                                state.tempQcur,         // output
                                state.wrapQ,            // input
                                offset, nEmbdHead,
                                config.rmsNormEps(), state.localSize)
                        .task("reductionFinalNormalization" + "_Qcur_" + i ,
                                    TransformerComputeKernelsLayered::reductionFinalNormalization, context,
                                    state.tempQcur,     // output
                                    nEmbdHead,
                                    config.rmsNormEps())
                            .task("mapContext" + "_Qcur_" + i,
                                    Qwen3Kernels::mapIndexInPlace, context,
                                    state.wrapQ,        // output
                                    weights.rms_att_QNormLayered[layerIndex],
                                    offset, nEmbdHead,
                                    state.tempQcur);
            }

//            unifiedLayer.task("dbg_copy_out_wrapQ",
//                    Qwen3Kernels::dbgCopy,
//                    state.wrapQ,
//                    state.dbgQ,
//                    state.positionHolder,
//                    layerIndex);
//            unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapQ);
//            unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapK);
//
            // Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            for (int i = 0; i < config.numberOfKeyValueHeads(); i++) {
                //rmsnorm(state.k, state.k, weights.attnKNorm[curLayer], i * nEmbdHead, nEmbdHead, config.rmsNormEps());
                int offset = i * nEmbdHead;
                unifiedLayer.task("reductionsOneBlock" + "_Kcur_" + i ,
                                Qwen3Kernels::reductionOneBlockWithLayerWithOffset,
                                context,
                                state.tempKcur,
                                state.wrapK,
                                offset,
                                nEmbdHead,
                                config.rmsNormEps(),
                                state.localSize)
                        // maybe in CPU ?
                        .task("reductionFinalNormalization" + "_Kcur_" + i ,
                                TransformerComputeKernelsLayered::reductionFinalNormalization,
                                context,
                                state.tempKcur,
                                nEmbdHead,
                                config.rmsNormEps())
                        .task("mapContext" + "_Kcur_" + i,
                                Qwen3Kernels::mapIndexInPlace,
                                context,
                                state.wrapK,
                                weights.rms_att_KNormLayered[layerIndex],
                                offset,
                                nEmbdHead,
                                state.tempKcur);
            }
            // dbg copy out
            //unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapQ);
            //unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapK);
            //unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapV);

            // rope rotation task graph
            unifiedLayer.task("ropeRotation",
                            Qwen3Kernels::ropeRotation,
                            context,
                            state.positionHolder,
                            state.wrapQ,            // out
                            state.wrapK,            // out
                            config.numberOfKeyValueHeads(),
                            nEmbdHead);

//            unifiedLayer.task("dbg_copy_out_wrapQ",
//                    Qwen3Kernels::dbgCopy,
//                    state.wrapQ,
//                    state.dbgQ,
//                    state.positionHolder,
//                    layerIndex);

            // dbg copy out
            //unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapQ);
            //unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapK);

            unifiedLayer.task("copyToCaches",
                    TransformerComputeKernelsLayered::copyToCache,
                    state.wrapKeyCache,         // out
                    state.wrapK,                // in
                    state.wrapValueCache,       // out
                    state.wrapV,                // in
                    state.positionHolder,
                    nEmbdGqa,
                    layerIndex,
                    config.contextLength());

//            unifiedLayer.task("dbg_copy_out_q",
//                    Qwen3Kernels::dbgCopy,
//                    state.wrapQ,
//                    state.dbgQ,
//                    state.positionHolder,
//                    layerIndex);
//
//            unifiedLayer.task("dbg_copy_out_keyCache",
//                    Qwen3Kernels::dbgCopy,
//                    state.wrapKeyCache,
//                    state.dbgKeyCache,
//                    state.positionHolder,
//                    layerIndex);
//
//            unifiedLayer.task("dbg_copy_out_ValueCache",
//                    Qwen3Kernels::dbgCopy,
//                    state.wrapValueCache,
//                    state.dbgValueCache,
//                    state.positionHolder,
//                    layerIndex);

            unifiedLayer.task("parallel-attention",
                    TransformerComputeKernelsLayered::processHeadsFlashAttention,
                    context,
                    state.wrapQ,
                    state.wrapKeyCache,
                    state.wrapValueCache,
                    state.wrapXb,               // out
                    config.numberOfHeads(),
                    nEmbdHead,
                    nEmbdGqa,
                    gqa,
                    state.positionHolder,
                    layerIndex,
                    config.contextLength());

//            unifiedLayer.task("dbg_copy_out_x",
//                    Qwen3Kernels::dbgCopy,
//                    state.wrapX,
//                    state.dbgX,
//                    state.positionHolder,
//                    layerIndex);
//
//            unifiedLayer.task("dbg_copy_out_xb",
//                    Qwen3Kernels::dbgCopy,
//                    state.wrapXb,
//                    state.dbgXb,
//                    state.positionHolder,
//                    layerIndex);

            //unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapXb);
            unifiedLayer.task("matmul1", Qwen3Kernels::matrixVectorGenericWithResidual,
                    context,
                    state.wrapXb,                           // vector
                    state.wrapX,                            // out, should be [1024]
                    weights.woLayered[layerIndex],          // matrix
                    nEmbdHeadK * config.numberOfHeads(),    // dim1 = 2048
                    config.dim(),                           // dim0 = 1024
                    LOCAL_WORK_GROUP_SIZE_ALLOC);

//            unifiedLayer.task("dbg_copy_out_x",
//                    Qwen3Kernels::dbgCopy,
//                    state.wrapX,
//                    state.dbgX,
//                    state.positionHolder,
//                    layerIndex);

            //unifiedLayer.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapX);
            unifiedLayer.task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                            context, state.tempFFN, state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                    .task("reductionFinalNormalizationFFN" , TransformerComputeKernelsLayered::reductionFinalNormalization, context, state.tempFFN,
                            config.dim(), config.rmsNormEps())
                    .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                            state.wrapX, weights.rms_ffn_weightLayered[layerIndex], state.tempFFN);

//            unifiedLayer.task("dbg_copy_out_xb",
//                    Qwen3Kernels::dbgCopy,
//                    state.wrapXb,
//                    state.dbgXb,
//                    state.positionHolder,
//                    layerIndex);

            unifiedLayer.task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context,
                            state.wrapXb,   state.wrapHb, weights.w1Layered[layerIndex], weights.w3Layered[layerIndex], config.dim(), config.hiddenDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("projectionTwo", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                            state.wrapHb, state.wrapX, weights.w2Layered[layerIndex], config.hiddenDim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                    //.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                    // dbg copy out
                    //.transferToHost(DataTransferMode.EVERY_EXECUTION, state.dbgQ, state.dbgKeyCache, state.dbgValueCache)
                    //.transferToHost(DataTransferMode.EVERY_EXECUTION, state.dbgX)//, state.dbgXb)
                    //.transferToHost(DataTransferMode.EVERY_EXECUTION, state.dbgValueCache)
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
                //.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.tempLogits,
                        state.wrapLogits
                )
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        context,
                        //state.wrapLogits,
                        weights.wclsHalfFloat,
                        weights.rms_final_weight_as_floatArray
                )
                .task("reductionsOneBlockLogits", TransformerComputeKernels::reductionOneBlockWithLayer,
                        context,
                        state.tempLogits,
                        state.wrapX,
                        config.dim(),
                        config.rmsNormEps(),
                        state.localSize)
//                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.tempLogits)
//                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapX)
//                .task("reductionFinalNormalizationLogits" , TransformerComputeKernelsLayered::reductionFinalNormalization, context, state.tempLogits,
//                        config.dim(), config.rmsNormEps())
                .task("mapContextLogits", TransformerComputeKernels::reductionOneBlock2WithLogits, context, state.wrapX,
                        weights.rms_final_weight_as_floatArray, state.tempLogits);
                //.transferToHost(DataTransferMode.EVERY_EXECUTION, state.tempLogits);
        logits = configureQuantizedMatrixVectorFinalWeight(logits);
        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        taskGraphs.add(logits.snapshot());

        return new Tuple2<>(taskGraphs, setupQwen3GridSchedulersLayeredNonNvidia());

    }

    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayeredNonNvidia() {
        return setupTornadoForwardPlanLayered();
    }

    private GridScheduler setupQwen3GridSchedulersLayeredNonNvidia() {
        GridScheduler gridScheduler = new GridScheduler();

        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
        rmsNormWorker.setGlobalWork(config.dim(), 1, 1);  // Set global work size to total dimension
        rmsNormWorker.setLocalWork(state.localSize, 1, 1);         // Set local work size to 256 (standard efficient size)

        int matmulQGlobal = nEmbdHeadK * config.numberOfHeads() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulQRowMajorWorker = new WorkerGrid1D(matmulQGlobal);
        matmulQRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        int matmulKVGlobal = nEmbdGqa * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulKVRowMajorWorker = new WorkerGrid1D(matmulKVGlobal);
        matmulKVRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        WorkerGrid curWorker = new WorkerGrid1D(nEmbdHead);  // mEmbdHead = 128
        curWorker.setGlobalWork(nEmbdHead, 1, 1);  // Set global work size to total dimension
        curWorker.setLocalWork(128, 1, 1);         // Set local work size to 256 (standard efficient size)

        int h = config.numberOfHeads();
        int ic = nEmbdHead / 2;
        WorkerGrid ropeWorker = new WorkerGrid2D(h, ic);
        ropeWorker.setGlobalWork(h, ic, 1);
        ropeWorker.setLocalWork(8, 1, 1);

        WorkerGrid copyToCachesWorker = new WorkerGrid1D(nEmbdGqa);
        copyToCachesWorker.setGlobalWork(nEmbdGqa, 1, 1);
        copyToCachesWorker.setLocalWork(128, 1, 1); // Set local work size to 32 (for copying to caches)

        // Parallel attention worker configuration
        WorkerGrid parallelAttentionWorker = new WorkerGrid1D(config.numberOfHeads()); // qwen ok
        // the global group work size is numberOfHeads * localWorkGroupSize, where the localWorkGroupSize is currently 4
        parallelAttentionWorker.setGlobalWork(config.numberOfHeads() * 8, 1, 1);
        parallelAttentionWorker.setLocalWork(8, 1, 1); // Set local work size to 4 (for parallel attention)

        int matmul1Global = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmul1Worker = new WorkerGrid1D(matmul1Global);
        matmul1Worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        int fusedFFNW1W3Global = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedFFNW1W3Worker = new WorkerGrid1D(fusedFFNW1W3Global);
        fusedFFNW1W3Worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        int projectionTwoGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid projectionTwoWorker = new WorkerGrid1D(projectionTwoGlobal);
        projectionTwoWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // Map workers to tasks
        gridScheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);
        for (int i = 0; i < config.numberOfLayers(); i++) {
            gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            //gridScheduler.addWorkerGrid("layer_" + i + ".reductionFinalNormalization", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".qmatmul", matmulQRowMajorWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".kmatmul", matmulKVRowMajorWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".vmatmul", matmulKVRowMajorWorker);

//            //int size = nEmbdHead;
//            for (int j = 0; j < config.numberOfHeads(); j++) {
////                int offset = j * nEmbdHead;
////                WorkerGrid qRmsReductionWorker = new WorkerGrid1D(size);
////                qRmsReductionWorker.setLocalWork(state.localSize, 1, 1);
//                gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock" + "_Qcur_" + j, curWorker);
//                //gridScheduler.addWorkerGrid("layer_" + i + ".reductionFinalNormalization" + "_Qcur_" + j, curWorker);
//                gridScheduler.addWorkerGrid("layer_" + i + ".mapContext" + "_Qcur_" + j, curWorker);
//            }
            // Create separate WorkerGrid for each head
            for (int j = 0; j < config.numberOfHeads(); j++) {
                WorkerGrid headWorker = new WorkerGrid1D(nEmbdHead);  // nEmbdHead = 128
                headWorker.setGlobalWork(nEmbdHead, 1, 1);  // Set global work size to total dimension
                headWorker.setLocalWork(128, 1, 1);

                gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock" + "_Qcur_" + j, headWorker);
                gridScheduler.addWorkerGrid("layer_" + i + ".mapContext" + "_Qcur_" + j, headWorker);
            }
            for (int j = 0; j < config.numberOfKeyValueHeads(); j++) {
                gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock" + "_Kcur_" + j, curWorker);
                //gridScheduler.addWorkerGrid("layer_" + i + ".reductionFinalNormalization" + "_Kcur_" + j, curWorker);
                gridScheduler.addWorkerGrid("layer_" + i + ".mapContext" + "_Kcur_" + j, curWorker);
            }

            gridScheduler.addWorkerGrid("layer_" + i + ".ropeRotation", ropeWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".matmul1", matmul1Worker);
            gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            //gridScheduler.addWorkerGrid("layer_" + i + ".reductionFinalNormalizationFFN", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", fusedFFNW1W3Worker);
            gridScheduler.addWorkerGrid("layer_" + i + ".projectionTwo", projectionTwoWorker);
        }

        int vocabSizeRowMajor = config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
        WorkerGrid vocabWorker = new WorkerGrid1D(vocabSizeRowMajor);
        vocabWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);

        gridScheduler.addWorkerGrid("logits.reductionsOneBlockLogits", rmsNormWorker);
        gridScheduler.addWorkerGrid("logits.mapContextLogits", rmsNormWorker);

        gridScheduler.addWorkerGrid("logits.projection", vocabWorker);

        return gridScheduler;
    }

}
