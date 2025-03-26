package com.example.tornadovm;

import com.example.aux.Tuple2;
import com.example.inference.engine.impl.Configuration;
import com.example.inference.engine.impl.Llama;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

import java.util.List;

public class TornadoVMMasterPlan {
    private final State state;
    private final Configuration config;
    private final Weights weights;
    List<ImmutableTaskGraph> taskGraphs;
    private GridScheduler scheduler;

    public TornadoVMMasterPlan(State state, Llama model) {
        TornadoVMLayerPlanner tornadoVMLayerPlanner = new TornadoVMLayerPlanner(state, model);
        Tuple2<List<ImmutableTaskGraph>, GridScheduler> tornadoVMPlan = tornadoVMLayerPlanner.setupTornadoForwardPlan();
        this.taskGraphs = tornadoVMPlan.getFirst();
        this.scheduler = tornadoVMPlan.getSecond();
        this.state = state;
        this.config = model.configuration();
        this.weights = model.weights();
    }

    public void tornadoVMForwardExecute(int position) {
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(
                // @formatter:off
                taskGraphs.get(0),
                taskGraphs.get(1),
                taskGraphs.get(2),
                taskGraphs.get(3),
                taskGraphs.get(4),
                taskGraphs.get(5),
                taskGraphs.get(6),
                taskGraphs.get(7))) {
            // @formatter:on
            state.positionAndLayer.set(0, position);

            // Update before execute (it an every copy in)
            executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

            for (int l = 0; l < config.numberOfLayers; l++) {
                System.out.println("");
                System.out.println("====== Start of layer ====== " + l);

                state.positionAndLayer.set(1, l); // Update before execute (it an every copy in)

                // Step 1: RMSNorm for attention
                executionPlan.withGraph(1).withGridScheduler(scheduler).execute();

                // Step 2: QKV Matmuls
                executionPlan.withGraph(2).withGridScheduler(scheduler).execute();

                // Step 3: RoPE rotation
                executionPlan.withGraph(3).withGridScheduler(scheduler).execute();

                // new shift by l
                //    // Calculate the offset based on layer, max sequence length, and position
                long offset = l * config.contextLength * kvDim + position * kvDim;

                System.out.println("Mapping memory regions at offset: " + offset);
                System.out.println("Key cache size: " + state.wrapKeyCache.getSize());
                System.out.println("K vector size: " + state.wrapK.getSize());

                System.out.println("Layer: " + l + ", Position: " + position);
                System.out.println("Dimensions - dim: " + config.dim + ", kvDim: " + kvDim + ", contextLength: " + config.contextLength);
                System.out.println("Calculated offset: " + offset);

                executionPlan.mapOnDeviceMemoryRegion(state.wrapKeyCache, state.wrapK, offset, 3, 4);
                executionPlan.mapOnDeviceMemoryRegion(state.wrapValueCache, state.wrapV, offset, 3, 4);

                // Step 4: Multi-head Attention (scores, softmax, weighted sum)
                executionPlan.withGraph(4).withGridScheduler(scheduler).execute();

                // Step 5: Feed-forward neural network
                executionPlan.withGraph(5).withGridScheduler(scheduler).execute();
                executionPlan.withAllGraphs();
                System.out.println("====== End of layer ====== " + l);
            }

            // Final RMSNorm
            executionPlan.withGraph(6).withGridScheduler(scheduler).execute();

            // Final projection to logits
            executionPlan.withGraph(7).withGridScheduler(scheduler).execute();

            // Copy results from TornadoVM buffers to state.logits
        } catch (TornadoExecutionPlanException e) {
            throw new RuntimeException(e);
        }
    }
}
