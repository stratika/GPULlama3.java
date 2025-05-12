package com.example;

import com.example.aot.AOT;
import com.example.aux.ChatFormat;
import com.example.core.model.tensor.FloatTensor;
import com.example.inference.CategoricalSampler;
import com.example.inference.Sampler;
import com.example.inference.ToppSampler;
import com.example.inference.engine.impl.Llama;
import com.example.inference.engine.impl.Options;
import com.example.loader.weights.ModelLoader;
import com.example.loader.weights.State;
import com.example.tornadovm.FloatArrayUtils;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.function.IntConsumer;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public class LlamaApp {
    // Configuration flags for hardware acceleration and optimizations
    public static final boolean USE_VECTOR_API = Boolean.parseBoolean(System.getProperty("llama.VectorAPI", "true"));   // Enable Java Vector API for CPU acceleration
    public static final boolean USE_AOT = Boolean.parseBoolean(System.getProperty("llama.AOT", "false"));               // Use Ahead-of-Time compilation
    public static final boolean USE_TORNADOVM = Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"));     // Use TornadoVM for GPU acceleration

    /**
     * Creates and configures a sampler for token generation based on specified parameters.
     *
     * <p>This method selects an appropriate sampling strategy for next-token prediction
     * in language model inference. It supports several sampling approaches:</p>
     *
     * <ul>
     *   <li>Greedy sampling (temperature = 0): Always selects the most probable token</li>
     *   <li>Temperature sampling: Adjusts probability distribution sharpness</li>
     *   <li>Top-p (nucleus) sampling: Considers only tokens comprising the top p probability mass</li>
     * </ul>
     *
     * <p>The method handles both {@link FloatTensor} and {@link FloatArray} logits types
     * to support both CPU and GPU execution paths.</p>
     *
     * @param vocabularySize The size of the model's vocabulary
     * @param temperature A value controlling randomness in sampling:
     *                   <ul>
     *                     <li>0.0f: No randomness (greedy sampling)</li>
     *                     <li>1.0f: Standard sampling from unmodified distribution</li>
     *                     <li>&lt;1.0f: More deterministic (sharper distribution)</li>
     *                     <li>&gt;1.0f: More random (flatter distribution)</li>
     *                   </ul>
     * @param topp The cumulative probability threshold for nucleus sampling (0.0-1.0).
     *            <ul>
     *              <li>Values ≤0 or ≥1: Disables top-p sampling</li>
     *              <li>Values in (0,1): Restricts sampling to tokens comprising the top p probability mass</li>
     *            </ul>
     * @param rngSeed Seed value for the random number generator to ensure reproducibility
     *
     * @return A configured {@link Sampler} that implements the selected sampling strategy
     *         and handles both tensor and array-based logits
     *
     * @throws IllegalArgumentException if logits are of an unsupported type
     */
    static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        Sampler sampler;
        if (temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            sampler = Sampler.TENSOR_ARGMAX; // Use TENSOR_ARGMAX instead of ARGMAX
        } else {
            // we sample from this distribution to get the next token
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
            Sampler innerSampler;
            // Determine whether to use top-p (nucleus) sampling
            if (topp <= 0 || topp >= 1) {
                // If topp is outside (0,1), use standard categorical sampling
                // This samples directly from the probability distribution
                innerSampler = new CategoricalSampler(rng);
            } else {
                // Use top-p (nucleus) sampling with the specified threshold
                // This restricts sampling to only the most likely tokens that
                // cumulatively comprise the top p probability mass
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }

            // Create a sampler that:
            // 1. Applies temperature scaling to the logits
            // 2. Converts logits to probabilities using softmax
            // 3. Delegates the actual sampling to the appropriate inner sampler
            sampler = logits -> {
                // Handle different logits formats to support both CPU and GPU paths
                if (logits instanceof FloatTensor) {
                    // For CPU path using FloatTensor
                    FloatTensor tensorLogits = (FloatTensor) logits;
                    // Apply temperature scaling - lower values make distribution more peaked
                    tensorLogits.divideInPlace(0, tensorLogits.size(), temperature);
                    // Convert logits to probabilities using softmax
                    tensorLogits.softmaxInPlace(0, tensorLogits.size());
                } else if (logits instanceof FloatArray) {
                    // For GPU path using FloatArray
                    FloatArray arrayLogits = (FloatArray) logits;
                    // Apply the same operations but using FloatArray-specific methods for TornadoVM data types
                    FloatArrayUtils.divideInPlace(arrayLogits, 0, arrayLogits.getSize(), temperature);
                    FloatArrayUtils.softmaxInPlace(arrayLogits, 0, arrayLogits.getSize());
                } else {
                    // If logits are neither FloatTensor nor FloatArray, throw an exception
                    throw new IllegalArgumentException("Unsupported logits type: " + (logits != null ? logits.getClass().getName() : "null"));
                }
                return innerSampler.sampleToken(logits);
            };
        }
        return sampler;
    }

    static void runInteractive(Llama model, Sampler sampler, Options options) {
        State state = null;
        List<Integer> conversationTokens = new ArrayList<>();
        ChatFormat chatFormat = new ChatFormat(model.tokenizer());
        conversationTokens.add(chatFormat.beginOfText);
        if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        int startPosition = 0;
        Scanner in = new Scanner(System.in);
        while (true) {
            System.out.print("> ");
            System.out.flush();
            String userText = in.nextLine();
            if (List.of("quit", "exit").contains(userText)) {
                break;
            }
            if (state == null) {
                state = model.createNewState();
            }
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, userText)));
            conversationTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            Set<Integer> stopTokens = chatFormat.getStopTokens();
            List<Integer> responseTokens = Llama.generateTokens(model, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(),
                    sampler, options.echo(), token -> {
                        if (options.stream()) {
                            if (!model.tokenizer().isSpecialToken(token)) {
                                System.out.print(model.tokenizer().decode(List.of(token)));
                            }
                        }
                    });
            // Include stop token in the prompt history, but not in the response displayed to the user.
            conversationTokens.addAll(responseTokens);
            startPosition = conversationTokens.size();
            Integer stopToken = null;
            if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                stopToken = responseTokens.getLast();
                responseTokens.removeLast();
            }
            if (!options.stream()) {
                String responseText = model.tokenizer().decode(responseTokens);
                System.out.println(responseText);
            }
            if (stopToken == null) {
                System.err.println("Ran out of context length...");
                break;
            }
        }
    }

    static void runInstructOnce(Llama model, Sampler sampler, Options options) {
        State state = model.createNewState();
        ChatFormat chatFormat = new ChatFormat(model.tokenizer());

        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(chatFormat.beginOfText);
        if (options.systemPrompt() != null) {
            promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        List<Integer> responseTokens;

        // Define the token consumer
        IntConsumer tokenConsumer = token -> {
            if (options.stream()) {
                if (!model.tokenizer().isSpecialToken(token)) {
                    System.out.print(model.tokenizer().decode(List.of(token)));
                }
            }
        };

        Set<Integer> stopTokens = chatFormat.getStopTokens();
        if (USE_TORNADOVM) {
            // Call generateTokensGPU without the token consumer parameter
            responseTokens = Llama.generateTokensGPU(model, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo());
            // Handle token output separately if needed
            // You might need to iterate through responseTokens and process them
            if (options.stream()) {
                for (Integer token : responseTokens) {
                    if (!model.tokenizer().isSpecialToken(token)) {
                        System.out.print(model.tokenizer().decode(List.of(token)));
                    }
                }
            }
        } else {
            // CPU path still uses the token consumer
            responseTokens = Llama.generateTokens(model, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), tokenConsumer);
        }

        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        if (!options.stream()) {
            String responseText = model.tokenizer().decode(responseTokens);
            System.out.println(responseText);
        }

        Llama.LastRunMetrics.printMetrics();

    }

    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Llama model;
        if (USE_AOT) {
            model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
        } else {
            model = ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true);
        }
        Sampler sampler = selectSampler(model.configuration().vocabularySize, options.temperature(), options.topp(), options.seed());
        if (options.interactive()) {
            runInteractive(model, sampler, options);
        } else {
            runInstructOnce(model, sampler, options);
        }
    }
}



