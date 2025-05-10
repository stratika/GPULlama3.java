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

    public FloatArray tornadoVMForwardExecuteLayered(int position) {
        // Execute the first TornadoVM graph (pre-processing) -> copy-in
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        state.positionHolder.set(0, position);
        for (int layer = 0; layer < config.numberOfLayers; layer++) {
            executionPlan.withGraph(layer+1).withGridScheduler(scheduler).execute();
        }

        // Execute the final TornadoVM graph (projection to logits)
        executionPlan.withGraph(config.numberOfLayers + 2 - 1).withGridScheduler(scheduler).execute();

        return state.wrapLogits;
    }

    /// Execute the forward pass of the LLaMA transformer model using TornadoVM acceleration
    /// just once to copy the data into the read-only data layer.
    public void forceCopyInReadOnlyDataLayered() {
        // Execute all TornadoVM graphs
        state.wrapX.init(0.0f);
        state.positionHolder.init(0);

        // Execute activation update graph
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        // Execute layer processing graphs
        for (int layer = 0; layer < config.numberOfLayers; layer++) {
            executionPlan.withGraph(layer + 1).withGridScheduler(scheduler).execute();
        }

        // Execute logits graph
        executionPlan.withGraph(config.numberOfLayers + 1).withGridScheduler(scheduler).execute();
    }

    /**
     * Frees the device memory allocated for the TornadoVM execution plan.
     * This method should be called when the execution plan is no longer needed
     * to release resources and avoid memory leaks.
     */
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }
}
