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
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
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

    private void checkMemoryUsage(String graphName) {
        System.out.printf("Before graph %s: KeyCache size=%d, Position=%d, Layer=%d\n",
                graphName, state.wrapKeyCache.getSize(),
                state.positionAndLayer.get(0), state.positionAndLayer.get(1));
    }

    private void logMemoryUsage(TornadoExecutionPlan executionPlan, String stage, int layer) {
        // Only log detailed memory usage in verbose mode or for critical operations
        if (true) {
            System.out.printf("Layer %d, %s: Memory usage = %.2f MB\n",
                    layer, stage, executionPlan.getCurrentDeviceMemoryUsage() / (1024.0 * 1024.0));
        }
    }

    private boolean isMemoryHighWatermark(TornadoExecutionPlan executionPlan, double thresholdPercent) {
        // Estimate total device memory (you may need to adjust this based on your device)
        long estimatedTotalMemory = 8L * 1024L * 1024L * 1024L;  // 8 GB for RTX 3070
        long currentUsage = executionPlan.getCurrentDeviceMemoryUsage();
        double usagePercent = (currentUsage * 100.0) / estimatedTotalMemory;

        boolean isHigh = usagePercent > thresholdPercent;
        if (isHigh) {
            System.out.printf("WARNING: High memory usage detected: %.2f%% (%.2f MB)\n",
                    usagePercent, currentUsage / (1024.0 * 1024.0));
        }
        return isHigh;
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
                taskGraphs.get(7),
                taskGraphs.get(8))
        ) {
            // @formatter:on
            state.positionAndLayer.set(0, position);

            // Update before execute (it an every copy in)
//            executionPlan.withGraph(0).withGridScheduler(scheduler).execute();
            executionPlan.withGraph(0).withGridScheduler(scheduler).execute();
//            executionPlan.forceCopyIn("rmsnorm",state.wrapX);

            for (int l = 0; l < config.numberOfLayers; l++) {
//                System.out.println("");
//                System.out.println("====== Start of layer ====== " + l);


                int layerOffset = l * config.contextLength * kvDim + position * kvDim;

                state.positionAndLayer.set(1, l);
                state.positionAndLayer.set(2, layerOffset);


                // Step 1: RMSNorm for attention
                executionPlan.withGraph(1).withGridScheduler(scheduler).execute();

                // Step 2: QKV Matmuls
                executionPlan.withGraph(2).withGridScheduler(scheduler).execute();

                // Step 3: RoPE rotation
                executionPlan.withGraph(3).withGridScheduler(scheduler).execute();

                executionPlan.withGraph(4).execute();

                // Step 4: Multi-head Attention (scores, softmax, weighted sum)
                 executionPlan.withGraph(5).withGridScheduler(scheduler).execute();

                // Step 5: Feed-forward neural network
                executionPlan.withGraph(6).withGridScheduler(scheduler).execute();

//                System.out.println("====== End of layer ====== " + l);
            }

            // Final RMSNorm
            executionPlan.withGraph(7).withGridScheduler(scheduler).execute();

            // Final projection to logits
            executionPlan.withGraph(8).withGridScheduler(scheduler).execute();

            // Copy results from TornadoVM buffers to state.logits
        } catch (TornadoExecutionPlanException e) {
            throw new RuntimeException(e);
        }
    }

    private void logMemoryUsage(TornadoExecutionResult executionResult, String stage, int layer) {
        long totalDeviceMemoryUsage = executionResult.getProfilerResult().getTotalDeviceMemoryUsage();
        double memoryInMB = totalDeviceMemoryUsage / (1024.0 * 1024.0);
        if (layer >= 0) {
            System.out.printf("Layer %d, %s: Total memory usage = %.2f MB\n",
                    layer, stage, memoryInMB);
        } else {
            System.out.printf("%s: Total memory usage = %.2f MB\n",
                    stage, memoryInMB);
        }
    }

//    public void tornadoVMForwardExecute(int position) {
//        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
//        try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(
//                // @formatter:off
//                taskGraphs.get(0),
//                taskGraphs.get(1),
//                taskGraphs.get(2),
//                taskGraphs.get(3),
//                taskGraphs.get(4),
//                taskGraphs.get(5),
//                taskGraphs.get(6),
//                taskGraphs.get(7))) {
//            // @formatter:on
//            state.positionAndLayer.set(0, position);
//
//            // Log initial memory usage
//            System.out.printf("Initial device memory usage: %.2f MB\n",
//                    executionPlan.getCurrentDeviceMemoryUsage() / (1024.0 * 1024.0));
//
//
//            // Update before execute (it an every copy in)
//            executionPlan.withGraph(0).withGridScheduler(scheduler).execute();
//
//            for (int l = 0; l < config.numberOfLayers; l++) {
//                System.out.println("");
//                System.out.println("====== Start of layer ====== " + l);
//
//
//                System.out.printf("Device memory usage after layer %d: %.2f MB\n",
//                        l, executionPlan.getCurrentDeviceMemoryUsage() / (1024.0 * 1024.0));
//
//                state.positionAndLayer.set(1, l); // Update before execute (it an every copy in)
////                int offset = l * config.contextLength * kvDim;
////                int layerOffset = offset  + position;
//
//                int offset = l * config.contextLength * kvDim;
//                int layerOffset = offset + position * kvDim; // Multiply position by kvDim
//
//                int cacheSize = state.wrapKeyCache.getSize();
//
//                if (layerOffset + kvDim > cacheSize) {
//                    System.err.printf("WARNING: Cache offset calculation would exceed bounds: " +
//                                    "layer=%d, position=%d, calculated offset=%d, cache size=%d\n",
//                            l, position, layerOffset, cacheSize);
//                    // Adjust to prevent out-of-bounds access
//                    layerOffset = Math.max(0, cacheSize - kvDim);
//                }
//
//
//                state.positionAndLayer.set(2, layerOffset);
//
//                // Step 1: RMSNorm for attention
//                executionPlan.withGraph(1).withGridScheduler(scheduler).execute();
//                logMemoryUsage(executionPlan, "after RMSNorm", l);
//
//
//                // Step 2: QKV Matmuls
//                executionPlan.withGraph(2).withGridScheduler(scheduler).execute();
//                logMemoryUsage(executionPlan, "after QKV Matmuls", l);
//
//                // Step 3: RoPE rotation
//                executionPlan.withGraph(3).withGridScheduler(scheduler).execute();
//                logMemoryUsage(executionPlan, "after RoPE", l);
//
//                System.out.printf("Layer=%d, Position=%d, KVDim=%d, Offset=%d, LayerOffset=%d, CacheSize=%d\n",
//                        l, position, kvDim, offset, layerOffset, state.wrapKeyCache.getSize());
//
//                executionPlan.withGraph(4).execute();
//                logMemoryUsage(executionPlan, "after Copy to Cache", l);
//
//                // Step 4: Multi-head Attention (scores, softmax, weighted sum)
//                executionPlan.withGraph(5).withGridScheduler(scheduler).execute();
//                logMemoryUsage(executionPlan, "after Attention", l);
//
//
//                // Step 5: Feed-forward neural network
//                executionPlan.withGraph(6).withGridScheduler(scheduler).execute();
//                logMemoryUsage(executionPlan, "after FFN", l);
//
//                executionPlan.withAllGraphs();
//                System.out.println("====== End of layer ====== " + l);
//            }
//
//            // Final RMSNorm
//            executionPlan.withGraph(7).withGridScheduler(scheduler).execute();
//
//            // Final projection to logits
//            executionPlan.withGraph(8).withGridScheduler(scheduler).execute();
//
//            // Copy results from TornadoVM buffers to state.logits
//        } catch (TornadoExecutionPlanException e) {
//            throw new RuntimeException(e);
//        }
//    }
}
