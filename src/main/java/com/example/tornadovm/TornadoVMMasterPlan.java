package com.example.tornadovm;

import com.example.aux.Tuple2;
import com.example.core.model.tensor.FloatTensor;
import com.example.inference.engine.impl.Configuration;
import com.example.inference.engine.impl.Llama;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;

import java.util.List;

public class TornadoVMMasterPlan {
    private final State state;
    private final Configuration config;
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
        this.executionPlan = new TornadoExecutionPlan(taskGraphs.toArray(new ImmutableTaskGraph[taskGraphs.size()]));
    }

    public FloatTensor tornadoVMForwardExecute(int position) {

        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        for (int layer = 0; layer < config.numberOfLayers; layer++) {
            int loff = layer * config.contextLength * config.kvDim;
            int layerOffsetForCaches = loff + position * config.kvDim;

            state.positionAndLayer.set(0, position);
            state.positionAndLayer.set(1, layer);
            state.positionAndLayer.set(2, layerOffsetForCaches);
            state.positionAndLayer.set(3, loff);

            executionPlan.withGraph(1).withGridScheduler(scheduler).execute();

        }

        executionPlan.withGraph(2).withGridScheduler(scheduler).execute();

        state.logits.asMemorySegment().copyFrom(state.wrapLogits.getSegment());
        for (int i = 0; i < 10; i++) {
            System.out.printf("wrapX[%d] = %f%n", i, state.wrapX.get(i));
        }

        int totalSize = state.logits.size();
        int step = Math.max(1, totalSize / 20);  // 1/20 = 5%

        for (int i = 0; i < totalSize; i += step) {
            System.out.printf("wrapLogits[%d] = %f%n", i, state.logits.getFloat(i));
        }

        return state.logits;
    }

    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }
}
