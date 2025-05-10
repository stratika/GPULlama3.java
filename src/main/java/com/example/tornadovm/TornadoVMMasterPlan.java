package com.example.tornadovm;

import com.example.aux.Tuple2;
import com.example.inference.engine.impl.Configuration;
import com.example.inference.engine.impl.Llama;
import com.example.loader.weights.State;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.List;

public class TornadoVMMasterPlan {
    private final State state;
    private final Configuration config;
    public GridScheduler scheduler;
    public TornadoExecutionPlan executionPlan;
    List<ImmutableTaskGraph> taskGraphs;
    public FloatArray wrapX;
    public TornadoVMMasterPlan(State state, Llama model) {
        TornadoVMLayerPlanner tornadoVMLayerPlanner = new TornadoVMLayerPlanner(state, model);
        Tuple2<List<ImmutableTaskGraph>, GridScheduler> tornadoVMPlan = tornadoVMLayerPlanner.setupTornadoForwardPlanLayered();
        this.taskGraphs = tornadoVMPlan.getFirst();
        this.scheduler = tornadoVMPlan.getSecond();
        this.state = state;
        this.config = model.configuration();
        this.executionPlan = new TornadoExecutionPlan(taskGraphs.toArray(new ImmutableTaskGraph[taskGraphs.size()]));
    }

    /**
     * Executes the forward pass of a LLaMA transformer model using TornadoVM acceleration.
     *This method processes the transformer layers in sequence for a particular token position in the context
     * window.
     *
     * <p>The execution happens in three phases:
     * <ol>
     *   <li>Initial token embedding lookup (already done before calling this method)</li>
     *   <li>Sequential processing through each transformer layer using TornadoVM</li>
     *   <li>Final projection to logits using TornadoVM</li>
     * </ol>
     *
     *
     * @param position
     *         The current position in the sequence being processed
     * @return FloatTensor containing the output logits for token prediction
     */
    public FloatArray tornadoVMForwardExecute(int position) {
        // Execute the first TornadoVM graph (pre-processing) -> copy-in
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        // executionPlan.copy-in(state.wrapX);
        // process each transformer layer sequentially
        for (int layer = 0; layer < config.numberOfLayers; layer++) {
            // Calculate offsets for KV cache access
            int loff = layer * config.contextLength * config.kvDim;
            int layerOffsetForCaches = loff + position * config.kvDim;

            // Set state information for the current position and layer
            state.positionAndLayer.set(0, position);
            state.positionAndLayer.set(1, layer);
            state.positionAndLayer.set(2, layerOffsetForCaches);
            state.positionAndLayer.set(3, loff);

            // Execute the layer-specific TornadoVM graph
            executionPlan.withGraph(1).withGridScheduler(scheduler).execute();
        }

        // Execute the final TornadoVM graph (projection to logits)
        executionPlan.withGraph(2).withGridScheduler(scheduler).execute();

        return state.wrapLogits;
    }

    public FloatArray tornadoVMForwardExecuteLayered(int position) {
        // Execute the first TornadoVM graph (pre-processing) -> copy-in
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        // executionPlan.copy-in(state.wrapX);
        // process each transformer layer sequentially
        state.positionAndLayer.set(0, position);
        for (int layer = 0; layer < config.numberOfLayers; layer++) {
            // Calculate offsets for KV cache access
            int loff = layer * config.contextLength * config.kvDim;
            int layerOffsetForCaches = loff + position * config.kvDim;

            // Set state information for the current position and layer
            state.positionAndLayer.set(1, layer);
            state.positionAndLayer.set(2, layerOffsetForCaches);
            state.positionAndLayer.set(3, loff);

            // Execute the layer-specific TornadoVM graph
            executionPlan.withGraph(layer+1).withGridScheduler(scheduler).execute();
        }

        // Execute the final TornadoVM graph (projection to logits)
        executionPlan.withGraph(config.numberOfLayers + 2 - 1).withGridScheduler(scheduler).execute();

        return state.wrapLogits;
    }

    // Force copy-in read-only weights
    public void forceCopyInReadOnlyData() {
        // Execute the first TornadoVM graph (pre-processing) -> copy-in
        state.wrapX.init(0.0f);
        state.positionAndLayer.init(0);
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();
        executionPlan.withGraph(1).withGridScheduler(scheduler).execute();
        executionPlan.withGraph(2).withGridScheduler(scheduler).execute();
    }

    public void forceCopyInReadOnlyDataLayered() {
        // Execute all TornadoVM graphs
        state.wrapX.init(0.0f);
        state.positionAndLayer.init(0);

        // Execute activation update graph
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        // Execute layer processing graphs
        for (int layer = 0; layer < config.numberOfLayers; layer++) {
            executionPlan.withGraph(layer + 1).withGridScheduler(scheduler).execute();
        }

        // Execute logits graph
        executionPlan.withGraph(config.numberOfLayers + 1).withGridScheduler(scheduler).execute();
    }
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }
}
