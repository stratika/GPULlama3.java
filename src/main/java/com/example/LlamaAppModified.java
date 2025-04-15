package com.example.debug;

import com.example.aot.AOT;
import com.example.inference.CategoricalSampler;
import com.example.inference.Sampler;
import com.example.inference.ToppSampler;
import com.example.inference.engine.impl.Llama;
import com.example.inference.engine.impl.Options;
import com.example.loader.weights.ModelLoader;

import java.io.IOException;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Debug application for Llama model
 */
public class LlamaAppModified {
    // Debug flags
    public static final boolean USE_VECTOR_API = Boolean.parseBoolean(System.getProperty("llama.VectorAPI", "true"));
    public static final boolean USE_AOT = Boolean.parseBoolean(System.getProperty("llama.AOT", "false"));
    public static final boolean COMPARE_OUTPUTS = Boolean.parseBoolean(System.getProperty("llama.CompareOutputs", "false"));
    public static final boolean SINGLE_TOKEN_DEBUG = Boolean.parseBoolean(System.getProperty("llama.SingleToken", "false"));
    public static final boolean DEBUG_LAYER = Boolean.parseBoolean(System.getProperty("llama.DebugLayer", "false"));
    public static final int DEBUG_LAYER_INDEX = Integer.parseInt(System.getProperty("llama.DebugLayerIndex", "0"));

    public static void main(String[] args) throws IOException {
        // Parse command line options
        Options options = Options.parseOptions(args);

        System.out.println("Loading model: " + options.modelPath());

        // Load the model
        Llama model;
        if (USE_AOT) {
            model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
        } else {
            model = ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true);
        }

        // Run the appropriate debug mode
        if (SINGLE_TOKEN_DEBUG) {
            debugSingleToken(model);
        } else if (DEBUG_LAYER) {
            debugLayer(model, DEBUG_LAYER_INDEX);
        } else if (COMPARE_OUTPUTS) {
            debugFullModel(model, options);
        } else {
            // Default to comparing outputs
            debugFullModel(model, options);
        }

        // Clean up
        com.example.debug.DebugHelper.shutdown();
    }

    /**
     * Debug a single token forward pass
     */
    private static void debugSingleToken(Llama model) {
        System.out.println("Running single token debug mode");

        // Choose a token to test
        int testToken = model.tokenizer().getSpecialTokens().getOrDefault("<|begin_of_text|>", 1);
        System.out.println("Using token: " + testToken);

        // Compare the implementations
        com.example.debug.DebugHelper.compareForwardImplementations(model, testToken, 0);
    }

    /**
     * Debug a specific layer
     */
    private static void debugLayer(Llama model, int layerIndex) {
        System.out.println("Running layer debug mode for layer " + layerIndex);

        // Choose a token to test
        int testToken = model.tokenizer().getSpecialTokens().getOrDefault("<|begin_of_text|>", 1);
        System.out.println("Using token: " + testToken);

        // Debug the specified layer
        com.example.debug.DebugHelper.debugLayer(model, testToken, 0, layerIndex);
    }

    /**
     * Debug the full model with a prompt
     */
    private static void debugFullModel(Llama model, Options options) {
        System.out.println("Running full model debug mode");
        System.out.println("Prompt: " + options.prompt());

        // Create a sampler with deterministic behavior
        Sampler sampler = createSampler(model.configuration().vocabularySize, 0.0f, 0.0f, options.seed());

        // Process first token of prompt for comparison
        String prompt = options.prompt();
        int firstToken = model.tokenizer().encode(prompt)[0];
        System.out.println("First token of prompt: " + firstToken);

        // Compare implementations with the first token
        com.example.debug.DebugHelper.compareForwardImplementations(model, firstToken, 0);
    }

    /**
     * Create a sampler for token generation
     */
    private static Sampler createSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        Sampler sampler;
        if (temperature == 0.0f) {
            // Greedy sampling - most predictable for debugging
            sampler = Sampler.ARGMAX;
        } else {
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
            Sampler innerSampler;

            if (topp <= 0 || topp >= 1) {
                innerSampler = new CategoricalSampler(rng);
            } else {
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }

            sampler = logits -> {
                logits.divideInPlace(0, logits.size(), temperature);
                logits.softmaxInPlace(0, logits.size());
                return innerSampler.sampleToken(logits);
            };
        }

        return sampler;
    }
}