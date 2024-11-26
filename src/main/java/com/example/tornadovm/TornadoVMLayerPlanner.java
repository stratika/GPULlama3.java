package com.example.tornadovm;

import com.example.aux.Tuple2;
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


    private Tuple2<TornadoExecutionPlan, GridScheduler> createTornadoExecutionPlan() {

        TaskGraph taskGraph;
        KernelContext context = new KernelContext();
        boolean isQ4Type = weights.wcls.toString().contains("Q4");

        taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapXFloat)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.wclsByteArray)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, configuration.dim)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, configuration.vocabularySize)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);

        if (isQ4Type) {
            taskGraph.task("t0", TornadoVMCompute::matmulTornadoQ4, context, weights.wclsByteArray, state.wrapXFloat, state.wrapLogits, configuration.dim);
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
                        weights.wclsByteArray, weights.rms_final_weight_as_floatArray, //
                        weights.woAsFloatArray[l], weights.w1AsFloatArray[l], //
                        weights.w2AFloatArray[l], weights.w3AFloatArray[l], //
                        configuration.vocabularySize, configuration.dim, //
                        configuration.rmsNormEps, configuration.hiddenDim, //
                        reduce
                ) //
                .task("matmul0", TornadoVMCompute::matrixVectorSimple, weights.woAsFloatArray[l], state.wrapXb, state.wrapXb2, configuration.dim, configuration.dim) //
                .task("addInPlace", TornadoVMCompute::addInPlace, state.wrapXb2, state.wrapXFloat) //
                .task("reduce", TornadoVMCompute::reduceSquareSums, context, state.wrapXb, reduce) //
                .task("sum", TornadoVMCompute::finalSum, context, reduce, configuration.dim, configuration.rmsNormEps) //
                .task("ns", TornadoVMCompute::normalizeAndScale, context, state.wrapXFloat, weights.rms_ffn_weight_as_floatArray[l], reduce, configuration.dim) //
                .task("matmul1", TornadoVMCompute::matrixVectorSimple, weights.w1AsFloatArray[l], state.wrapXb, state.wrapHb, configuration.hiddenDim, configuration.dim) //
                .task("matmul3", TornadoVMCompute::matrixVectorSimple, weights.w3AFloatArray[l], state.wrapXb, state.wrapXb2, configuration.hiddenDim, configuration.dim) //
                .task("mapInPlace", TornadoVMCompute::mapInPlace, state.wrapHb) //
                .task("multInPlace", TornadoVMCompute::multiplyInPlace, state.wrapHb, state.wrapHb2) //
                .task("matmul2", TornadoVMCompute::matrixVectorSimple, weights.w2AFloatArray[l], state.wrapXb, state.wrapXb, configuration.dim, configuration.hiddenDim) //
                .task("addInPlace2", TornadoVMCompute::addInPlace, state.wrapXb, state.wrapXFloat) //
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
