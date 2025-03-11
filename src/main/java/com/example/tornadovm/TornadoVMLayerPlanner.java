package com.example.tornadovm;

import com.example.aux.Tuple2;
import com.example.core.model.tensor.FloatTensor;
import com.example.inference.engine.impl.Configuration;
import com.example.inference.engine.impl.Llama;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.FloatBuffer;
import java.util.ArrayList;

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
                .task("ns", TornadoVMCompute::normalizeAndScale, context, state.wrapXFloat, weights.rms_final_weight_as_floatArray, reduce, configuration.dim)
                .task("matmul-1", TornadoVMCompute::matmulTornadoQ8, context, weights.wclsByteArray, state.wrapXFloat, state.wrapLogits, configuration.dim)
                .task("matmul-2", TornadoVMCompute::matmulTornadoQ8, context, weights.wclsByteArray, state.wrapXFloat, state.wrapLogits, configuration.dim)
                .task("matmul-3", TornadoVMCompute::matmulTornadoQ8, context, weights.wclsByteArray, state.wrapXFloat, state.wrapLogits, configuration.dim)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits, state.wrapXFloat);



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

        return new Tuple2<>(taskGraph, gridScheduler);
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
}
