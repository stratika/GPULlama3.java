package com.example.tornadovm;

import com.example.aux.Tuple2;
import com.example.inference.engine.impl.Configuration;
import com.example.inference.engine.impl.Llama;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;

import java.util.List;

public class TornadoVMMasterPlan {
    private final State state;
    private final Configuration config;
    private final Weights weights;
    List<ImmutableTaskGraph> taskGraphs;
    private GridScheduler scheduler;
    private TornadoExecutionPlan executionPlan;

    public TornadoVMMasterPlan(State state, Llama model) {
        TornadoVMLayerPlanner tornadoVMLayerPlanner = new TornadoVMLayerPlanner(state, model);
        Tuple2<List<ImmutableTaskGraph>, GridScheduler> tornadoVMPlan = tornadoVMLayerPlanner.setupTornadoForwardPlan();
        this.taskGraphs = tornadoVMPlan.getFirst();
        this.scheduler = tornadoVMPlan.getSecond();
        this.state = state;
        this.config = model.configuration();
        this.weights = model.weights();
        this.executionPlan = new TornadoExecutionPlan(taskGraphs.toArray(new ImmutableTaskGraph[taskGraphs.size()]));
    }

    private void checkMemoryUsage(String graphName) {
        System.out.printf("Before graph %s: KeyCache size=%d, Position=%d, Layer=%d\n", graphName, state.wrapKeyCache.getSize(), state.positionAndLayer.get(0), state.positionAndLayer.get(1));
    }

    private void logMemoryUsage(TornadoExecutionPlan executionPlan, String stage, int layer) {
        // Only log detailed memory usage in verbose mode or for critical operations
        if (true) {
            System.out.printf("Layer %d, %s: Memory usage = %.2f MB\n", layer, stage, executionPlan.getCurrentDeviceMemoryUsage() / (1024.0 * 1024.0));
        }
    }


    public void tornadoVMForwardExecute(int position) {
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;

        // @formatter:on
        state.positionAndLayer.set(0, position);

        // Update before execute (it an every copy in)

        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        for (int l = 0; l < config.numberOfLayers; l++) {

            int layerOffset = l * config.contextLength * kvDim + position * kvDim;

            state.positionAndLayer.set(1, l);
            state.positionAndLayer.set(2, layerOffset);

            // Step 1: RMSNorm for attention
            executionPlan.withGraph(1).withGridScheduler(scheduler).execute();

            //                System.out.println("====== End of layer ====== " + l);
        }

        // Final RMSNorm and Logits
        executionPlan.withGraph(2).withGridScheduler(scheduler).execute();

        // Copy results from TornadoVM buffers to state.logits
    }

    private void logMemoryUsage(TornadoExecutionResult executionResult, String stage, int layer) {
        long totalDeviceMemoryUsage = executionResult.getProfilerResult().getTotalDeviceMemoryUsage();
        double memoryInMB = totalDeviceMemoryUsage / (1024.0 * 1024.0);
        if (layer >= 0) {
            System.out.printf("Layer %d, %s: Total memory usage = %.2f MB\n", layer, stage, memoryInMB);
        } else {
            System.out.printf("%s: Total memory usage = %.2f MB\n", stage, memoryInMB);
        }
    }

    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }
}
