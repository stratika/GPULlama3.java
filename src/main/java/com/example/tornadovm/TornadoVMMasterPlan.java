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


    public void tornadoVMForwardExecute(int position) {
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;

        state.positionAndLayer.set(0, position);

        // Update before execute (it an every copy in)
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        for (int l = 0; l < config.numberOfLayers; l++) {
            state.positionAndLayer.set(1, l);

            int loff = l * config.contextLength * kvDim;
            state.positionAndLayer.set(3, loff);
            int layerOffsetForCaches =  loff + position * kvDim;
            state.positionAndLayer.set(2, layerOffsetForCaches);

            // Layer taskgraph
            executionPlan.withGraph(1).withGridScheduler(scheduler).execute();
        }

        // Final RMSNorm and Logits
        executionPlan.withGraph(2).withGridScheduler(scheduler).execute();

    }

    public void tornadoVMForwardExecuteLayer(int position, int ll) {
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;

        state.positionAndLayer.set(0, position);

        // Update before execute (it an every copy in)
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        for (int l = 0; l < ll; l++) {
            state.positionAndLayer.set(1, l);

            int loff = l * config.contextLength * kvDim;
            state.positionAndLayer.set(3, loff);
            int layerOffsetForCaches =  loff + position * kvDim;
            state.positionAndLayer.set(2, layerOffsetForCaches);

            // Layer taskgraph
            executionPlan.withGraph(1).withGridScheduler(scheduler).execute();
        }

        // Final RMSNorm and Logits
        executionPlan.withGraph(2).withGridScheduler(scheduler).execute();

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
