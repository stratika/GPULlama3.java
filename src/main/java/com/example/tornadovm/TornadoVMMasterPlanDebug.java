package com.example.tornadovm;

import com.example.aux.Tuple2;
import com.example.inference.engine.impl.Configuration;
import com.example.inference.engine.impl.Llama;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;

import java.lang.foreign.MemorySegment;
import java.util.List;

public class TornadoVMMasterPlanDebug {
    private final State state;
    private final Configuration config;
    private final Weights weights;
    List<ImmutableTaskGraph> taskGraphs;
    private GridScheduler scheduler;
    private TornadoExecutionPlan executionPlan;

    public TornadoVMMasterPlanDebug(State state, Llama model) {
        TornadoVMLayerPlannerDebug tornadoVMLayerPlanner = new TornadoVMLayerPlannerDebug(state, model);
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
            int layerOffsetForCaches = loff + position * kvDim;
            state.positionAndLayer.set(2, layerOffsetForCaches);

            // Layer taskgraph
            executionPlan.withGraph(1).withGridScheduler(scheduler).execute();
        }

        // Final RMSNorm and Logits
        executionPlan.withGraph(2).withGridScheduler(scheduler).execute();

    }

    public void tornadoVMForwardExecuteLayer(int position, int ll, int token, Llama model) {
        Configuration config = model.configuration();

        Weights weights = model.weights();
        int dim = config.dim;

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        MemorySegment.copy(state.x.asMemorySegment(), 0, state.wrapX.getSegment(), 0, dim * 4);

        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;

        //        [x] Validated -> before results are correct
        //        System.out.println("\n==== TornadoVM Input State ====");
        //        System.out.println("First 5 values of x tensor:");
        //        for (int i = 0; i < 15; i++) {
        //            System.out.printf("x[%d] = %f%n", i, state.wrapX.get(i));
        //        }

        state.positionAndLayer.set(0, position);

        // Update before execute (it an every copy in)
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        //        [x] Validated -> after results are correct
        //        System.out.println("\n==== POST 1ST TAST State ====");
        //        System.out.println("First 10 values of x tensor:");
        //        for (int i = 0; i < 15; i++) {
        //            System.out.printf("wrap[%d] = %f%n", i, state.wrapX.get(i));
        //        }

        for (int l = 0; l < ll; l++) {
            state.positionAndLayer.set(1, l);

            int loff = l * config.contextLength * kvDim;
            state.positionAndLayer.set(3, loff);
            int layerOffsetForCaches = loff + position * kvDim;
            state.positionAndLayer.set(2, layerOffsetForCaches);

            // Layer taskgraph
            executionPlan.withGraph(1).withGridScheduler(scheduler).execute();
        }
        System.out.println("\n==== Intermediate State ====");
        System.out.println("First 5 values of x tensor:");
        for (int i = 0; i < 5; i++) {
            System.out.printf("wrap[%d] = %f%n", i, state.wrapX.get(i));
        }

        System.out.println("\nFirst 5 values of xb tensor:");
        for (int i = 0; i < 5; i++) {
            System.out.printf("wrapXb[%d] = %f%n", i, state.wrapXb.get(i));
        }

        System.out.println("\nFirst 5 values of q tensor:");
        for (int i = 0; i < 5; i++) {
            System.out.printf("wrapQ[%d] = %f%n", i, state.wrapQ.get(i));
        }

        // Final RMSNorm and Logits
        //        executionPlan.withGraph(2).withGridScheduler(scheduler).execute();

    }

    /**
     * Executes the TornadoVM forward pass for a layer with validation at each stage
     *
     * @param position
     *         Current position in the sequence
     * @param ll
     *         Number of layers to process
     * @param token
     *         Current token
     * @param model
     *         The LLama model instance
     */
    public void tornadoVMForwardExecuteLayerWithValidation(int position, int ll, int token, Llama model) {
        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;

        // Copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);
        MemorySegment.copy(state.x.asMemorySegment(), 0, state.wrapX.getSegment(), 0, dim * 4);

        //        System.out.println("\n==== INITIAL STATE (Before execution) ====");
        //        System.out.println("First 5 values of x tensor:");
        //        for (int i = 0; i < 5; i++) {
        //            System.out.printf("x[%d] = %f%n", i, state.wrapX.get(i));
        //        }
        System.out.println("\n==== Tornado Debug Start ====");

        state.positionAndLayer.set(0, position);

        // Execute Graph 0: Buffer Initialization
        System.out.println("\n==== EXECUTING GRAPH 0: Buffer Initialization ====");
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        for (int l = 0; l < ll; l++) {
            System.out.println("\n==== PROCESSING LAYER " + l + " ====");
            state.positionAndLayer.set(1, l);

            int loff = l * config.contextLength * kvDim;
            state.positionAndLayer.set(3, loff);
            int layerOffsetForCaches = loff + position * kvDim;
            state.positionAndLayer.set(2, layerOffsetForCaches);

            // Execute Graph 1: RMS Norm
            System.out.println("\n==== EXECUTING GRAPH 1: unified ====");
            executionPlan.withGraph(1).withGridScheduler(scheduler).execute();

            System.out.println("After RMS Norm - First 5 values of xb tensor:");
            for (int i = 0; i < 10; i++) {
                System.out.printf("wrapXb[%d] = %f%n", i, state.wrapXb.get(i));
            }

            // Execute Graph 2: QKV Matmuls
            System.out.println("\n==== EXECUTING GRAPH 2: QKV Matmuls ====");
            executionPlan.withGraph(2).withGridScheduler(scheduler).execute();

            System.out.println("After QKV Matmuls - First 15 values of q, k, v tensors:");
            for (int i = 0; i < 15; i++) {
                System.out.printf("wrapQ[%d] = %f, wrapK[%d] = %f, wrapV[%d] = %f%n", i, state.wrapQ.get(i), i, state.wrapK.get(i), i, state.wrapV.get(i));
            }
            //            }
            //
            // Execute Graph 3: RoPE Rotation
            System.out.println("\n==== EXECUTING GRAPH 3: RoPE Rotation ====");
            executionPlan.withGraph(3).withGridScheduler(scheduler).execute();
            //
            System.out.println("After RoPE - First 5 values of q, k tensors (rotated):");
            for (int i = 0; i < 15; i++) {
                System.out.printf("wrapQ[%d] = %f, wrapK[%d] = %f%n", i, state.wrapQ.get(i), i, state.wrapK.get(i));
            }

            // Execute Graph 4: Copy to Caches
            System.out.println("\n==== EXECUTING GRAPH 4: Copy to Caches ====");
            executionPlan.withGraph(4).withGridScheduler(scheduler).execute();

            System.out.println("After Copy to Caches - First 5 values of key/value caches:");
            int cacheOffset = layerOffsetForCaches;
            for (int i = 0; i < 15; i++) {
                System.out.printf("keyCache[%d] = %f, valueCache[%d] = %f%n", cacheOffset + i, state.wrapKeyCache.get(cacheOffset + i), cacheOffset + i, state.wrapValueCache.get(cacheOffset + i));
            }
            //
            //            // Execute Graph 5: Multi-head Attention
            System.out.println("\n==== EXECUTING GRAPH 5: Multi-head Attention ====");
            executionPlan.withGraph(5).withGridScheduler(scheduler).execute();

            System.out.println("After Attention - First 5 values of xb tensor (attention output):");
            for (int i = 0; i < 15; i++) {
                System.out.printf("wrapXb[%d] = %f%n", i, state.wrapXb.get(i));
            }
            //
            // Execute Graph 6: Attention Output Processing
            System.out.println("\n==== EXECUTING GRAPH 6: Attention Output Processing ====");
            executionPlan.withGraph(6).withGridScheduler(scheduler).execute();

            System.out.println("After Attention Output - First 5 values of x tensor (after residual):");
            for (int i = 0; i < 15; i++) {
                System.out.printf("wrapX[%d] = %f, wrapXb2[%d] = %f%n", i, state.wrapX.get(i), i, state.wrapXb2.get(i));
            }
            //
            // Execute Graph 7: FFN Part 1 (Norm)
            System.out.println("\n==== EXECUTING GRAPH 7: FFN Norm ====");
            executionPlan.withGraph(7).withGridScheduler(scheduler).execute();

            System.out.println("After FFN Norm - First 5 values of xb tensor:");
            for (int i = 0; i < 15; i++) {
                System.out.printf("wrapXb[%d] = %f%n", i, state.wrapXb.get(i));
                //                rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

            }
            //
            // Execute Graph 8: FFN Part 2 (Projections)
            System.out.println("\n==== EXECUTING GRAPH 8: FFN Projections ====");
            executionPlan.withGraph(8).withGridScheduler(scheduler).execute();

            System.out.println("After FFN Projections - First 5 values of hb, hb2 tensors:");
            for (int i = 0; i < 15; i++) {
                System.out.printf("wrapHb[%d] = %f, wrapHb2[%d] = %f%n", i, state.wrapHb.get(i), i, state.wrapHb2.get(i));
            }
            //
            // Execute Graph 9: FFN Part 3 (Activation)
            System.out.println("\n==== EXECUTING GRAPH 9: FFN Activation ====");
            executionPlan.withGraph(9).withGridScheduler(scheduler).execute();

            System.out.println("After FFN Activation - First 5 values of hb tensor (after SiLU):");
            for (int i = 0; i < 15; i++) {
                System.out.printf("wrapHb[%d] = %f%n", i, state.wrapHb.get(i));
            }

            //             Execute Graph 10: FFN Part 4 (Final Projections)
            System.out.println("\n==== EXECUTING GRAPH 10: FFN Final ====");
            executionPlan.withGraph(10).withGridScheduler(scheduler).execute();

            System.out.println("After FFN Final - First 5 values of x tensor (after residual):");
            for (int i = 0; i < 15; i++) {
                System.out.printf("wrapX[%d] = %f, wrapXb[%d] = %f%n", i, state.wrapX.get(i), i, state.wrapXb.get(i));
            }

            System.out.println("\n==== End PROCESSING LAYER " + l + " ====");

        }
        System.out.println("\n==== LOGITS " + " ====");
        System.out.println("\n==== EXECUTING GRAPH 10: FFN Final ====");
        executionPlan.withGraph(11).withGridScheduler(scheduler).execute();
        executionPlan.withGraph(12).withGridScheduler(scheduler).execute();

        System.out.println("After TOKEN print logits first 45:");

        for (int i = 0; i < 10; i++) {
            System.out.printf("wrapX[%d] = %f%n", i, state.wrapX.get(i));
        }

        for (int i = 0; i < state.wrapLogits.getSize(); i++) {
            System.out.printf("wrapLogits[%d] = %f%n", i, state.wrapLogits.get(i));
        }

        System.out.println("\n==== Tornado Debug End ====");

    }

    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }
}
