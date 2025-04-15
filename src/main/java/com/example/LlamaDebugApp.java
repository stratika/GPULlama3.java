package com.example;

import com.example.aot.AOT;
import com.example.aux.ChatFormat;
import com.example.aux.Timer;
import com.example.core.model.tensor.FloatTensor;
import com.example.inference.CategoricalSampler;
import com.example.inference.Sampler;
import com.example.inference.ToppSampler;
import com.example.inference.engine.impl.Llama;
import com.example.inference.engine.impl.Options;
import com.example.loader.weights.ModelLoader;
import com.example.loader.weights.State;
import com.example.tornadovm.TornadoVMCompute;
import com.example.tornadovm.TornadoVMMasterPlanDebug;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Debug version of LlamaApp that compares Java and TornadoVM execution paths
 * for validation and debugging purposes.
 */
public class LlamaDebugApp {
    public static final boolean USE_VECTOR_API = Boolean.parseBoolean(System.getProperty("llama.VectorAPI", "true"));
    public static final boolean USE_AOT = Boolean.parseBoolean(System.getProperty("llama.AOT", "false"));

    static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        Sampler sampler;
        if (temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            sampler = Sampler.ARGMAX;
        } else {
            // we sample from this distribution to get the next token
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
            Sampler innerSampler;
            if (topp <= 0 || topp >= 1) {
                // simply sample from the predicted probability distribution
                innerSampler = new CategoricalSampler(rng);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }
            sampler = logits -> {
                // apply the temperature to the logits
                logits.divideInPlace(0, logits.size(), temperature);
                // apply softmax to the logits to get the probabilities for next token
                logits.softmaxInPlace(0, logits.size());
                return innerSampler.sampleToken(logits);
            };
        }
        return sampler;
    }

    /**
     * Compare the outputs of Java and TornadoVM implementations for a single forward pass
     */
    static void runValidationComparison(Llama model) {
        // Create two identical states for Java and TornadoVM paths
        State javaState = model.createNewState();
        State tornadoState = model.createNewState();

        // Get initial token (usually the BOS token)
        int token = javaState.latestToken;
        System.out.println("Starting validation with token: " + token);

        int position = 0; // Start at position 0

        System.out.println("\n==== Running Java implementation ====");
        try (var timer = Timer.log("Java forward")) {
            Llama.forwardJavaDebug(model, javaState, token, position, 1);
        }

        // Store Java logits for comparison
        float[] javaLogits = new float[javaState.logits.size()];
        for (int i = 0; i < javaLogits.length; i++) {
            javaLogits[i] = javaState.logits.getFloat(i);
        }

        System.out.println("\n==== Running TornadoVM Debug implementation ====");
        TornadoVMMasterPlanDebug tornadoDebug = new TornadoVMMasterPlanDebug(tornadoState, model);
        try (var timer = Timer.log("TornadoVM forward")) {
            // This will only execute layer 0 for debugging
            tornadoDebug.tornadoVMForwardExecuteLayer(position, 1, token, model);
        } finally {
//            tornadoDebug.freeTornadoExecutionPlan();
        }

        // Now validate the results
        System.out.println("\n==== Validation Results ====");
//        validateStates(javaState, tornadoState);
    }

    /**
     * Validate the states between Java and TornadoVM implementations
     */
    private static void validateStates(State javaState, State tornadoState) {
        // First compare basic tensors (x, xb, xb2, q, k, v)
        System.out.println("Validating x tensor:");
        compareTensors(javaState.x, tornadoState.wrapX, 5);

        System.out.println("Validating xb tensor:");
        compareTensors(javaState.xb, tornadoState.wrapXb, 5);

        System.out.println("Validating q tensor:");
        compareTensors(javaState.q, tornadoState.wrapQ, 5);

        System.out.println("Validating k tensor:");
        compareTensors(javaState.k, tornadoState.wrapK, 5);

        System.out.println("Validating v tensor:");
        compareTensors(javaState.v, tornadoState.wrapV, 5);

        // If testing a full run that includes logits, also compare them
//        if (tornadoState.logits.size() > 0 && javaState.logits.size() > 0) {
//            System.out.println("Validating logits tensor:");
//            compareTensors(javaState.logits, tornadoState.logits, 5);
//        }
    }

    /**
     * Compare two tensors and print statistics about their differences
     */
    private static void compareTensors(FloatTensor a, FloatArray b, int printItems) {
        if (a.size() != b.getSize()) {
            System.out.println("ERROR: Tensor sizes do not match! A: " + a.size() + ", B: " + b.getSize());
            return;
        }

        float maxDiff = 0.0f;
        float totalDiff = 0.0f;
        int diffCount = 0;

        // Print first few items
        System.out.println("Sample values comparison:");
        for (int i = 0; i < Math.min(printItems, a.size()); i++) {
            float aVal = a.getFloat(i);
            float bVal = b.get(i);
            float diff = Math.abs(aVal - bVal);
            System.out.printf("  [%d] A: %f, B: %f, Diff: %f%n", i, aVal, bVal, diff);
        }

        // Calculate statistics
        for (int i = 0; i < a.size(); i++) {
            float aVal = a.getFloat(i);
            float bVal = b.get(i);
            float diff = Math.abs(aVal - bVal);

            totalDiff += diff;
            maxDiff = Math.max(maxDiff, diff);

            if (diff > 1e-5) {
                diffCount++;
            }
        }

        // Report statistics
        System.out.println("Comparison stats:");
        System.out.println("  Max difference: " + maxDiff);
        System.out.println("  Average difference: " + (totalDiff / a.size()));
        System.out.println("  Significant differences (>1e-5): " + diffCount + " out of " + a.size() +
                " (" + (100.0 * diffCount / a.size()) + "%)");
        System.out.println();
    }

    /**
     * Run the full model in interactive mode, comparing Java and TornadoVM implementations
     */
    static void runComparisonInteractive(Llama model, Sampler sampler, Options options) {
        State javaState = model.createNewState();
        State tornadoState = model.createNewState();

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

            // Process user input for both implementations
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, userText)));
            conversationTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            Set<Integer> stopTokens = chatFormat.getStopTokens();

            // Run both implementations in parallel or sequentially
            System.out.println("\nRunning Java implementation...");
            List<Integer> javaResponseTokens = runImplementation(model, javaState, startPosition,
                    conversationTokens.subList(startPosition, conversationTokens.size()),
                    stopTokens, options, sampler, false);

            System.out.println("\nRunning TornadoVM implementation...");
            // Force TornadoVM
            boolean oldTornadoSetting = TornadoVMCompute.TORNADOVM;
            System.setProperty("use.tornadovm", "true");
            List<Integer> tornadoResponseTokens = runImplementation(model, tornadoState, startPosition,
                    conversationTokens.subList(startPosition, conversationTokens.size()),
                    stopTokens, options, sampler, true);

            // Reset TORNADOVM flag
            System.setProperty("use.tornadovm", String.valueOf(oldTornadoSetting));

            // Compare responses
            System.out.println("\nComparison of responses:");
            compareResponses(javaResponseTokens, tornadoResponseTokens, model);

            // Update conversation state
            conversationTokens.addAll(javaResponseTokens);
            startPosition = conversationTokens.size();
        }
    }

    /**
     * Compare token sequences between Java and TornadoVM implementations
     */
    private static void compareResponses(List<Integer> javaTokens, List<Integer> tornadoTokens, Llama model) {
        System.out.println("Java response length: " + javaTokens.size());
        System.out.println("TornadoVM response length: " + tornadoTokens.size());

        int matchCount = 0;
        int minLength = Math.min(javaTokens.size(), tornadoTokens.size());

        for (int i = 0; i < minLength; i++) {
            if (javaTokens.get(i).equals(tornadoTokens.get(i))) {
                matchCount++;
            }
        }

        System.out.println("Matching tokens: " + matchCount + " out of " + minLength +
                " (" + (100.0 * matchCount / minLength) + "%)");

        // Show a few examples of differences
        int diffCount = 0;
        System.out.println("\nSample differences:");
        for (int i = 0; i < minLength && diffCount < 5; i++) {
            if (!javaTokens.get(i).equals(tornadoTokens.get(i))) {
                System.out.println("Position " + i + ":");
                System.out.println("  Java: " + javaTokens.get(i) + " (" +
                        model.tokenizer().decode(List.of(javaTokens.get(i))) + ")");
                System.out.println("  TornadoVM: " + tornadoTokens.get(i) + " (" +
                        model.tokenizer().decode(List.of(tornadoTokens.get(i))) + ")");
                diffCount++;
            }
        }
    }

    /**
     * Run a specific implementation (Java or TornadoVM)
     */
    private static List<Integer> runImplementation(Llama model, State state, int startPosition,
            List<Integer> promptTokens, Set<Integer> stopTokens, Options options, Sampler sampler,
            boolean useTornadoVM) {

        // Set TornadoVM flag
        boolean oldSetting = TornadoVMCompute.TORNADOVM;
        if (useTornadoVM) {
            System.setProperty("use.tornadovm", "true");
        } else {
            System.setProperty("use.tornadovm", "false");
        }

        // Generate tokens
        List<Integer> responseTokens = Llama.generateTokens(model, state, startPosition, promptTokens,
                stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
                    if (options.stream() && !model.tokenizer().isSpecialToken(token)) {
                        System.out.print(model.tokenizer().decode(List.of(token)));
                    }
                });

        // If not streaming, print the entire response
        if (!options.stream()) {
            String responseText = model.tokenizer().decode(responseTokens);
            System.out.println(responseText);
        }

        // Reset TornadoVM setting
        System.setProperty("use.tornadovm", String.valueOf(oldSetting));

        return responseTokens;
    }

    /**
     * Run a single forward operation, useful for detailed step-by-step debugging
     */
    static void runSingleForwardOperation(Llama model, Options options) {
        State javaState = model.createNewState();
        int token = javaState.latestToken;
        int position = 0;

        // Create debug master plan
        TornadoVMMasterPlanDebug debugPlan = new TornadoVMMasterPlanDebug(javaState, model);

        // Perform the Java version for baseline comparison
        System.out.println("==== Running Java reference implementation ====");
        FloatTensor javaLogits;
        try (var timer = Timer.log("Java forward")) {
            javaLogits = Llama.forwardJavaDebug(model, javaState, token, position, 1);
        }

        // Reset state for TornadoVM run
        State tornadoState = model.createNewState();

        // Run TornadoVM implementation for a single layer
        System.out.println("\n==== Running TornadoVM Debug implementation (Layer 0 only) ====");
        try (var timer = Timer.log("TornadoVM forward")) {
            debugPlan.tornadoVMForwardExecuteLayerWithValidation(position, 1, token, model);
        } finally {
//            debugPlan.freeTornadoExecutionPlan();
        }

        // Compare outputs - for a complete run we would compare logits
        // For a partial run (layer 0 only), we need to check intermediate states
//
    }

    public static void main(String[] args) throws IOException {
        // Parse command-line arguments
        Options options = Options.parseOptions(args);

        // Add debug option parsing
        boolean runDebugValidation = false;
        boolean runSingleForward = true;
        boolean runComparison = false;


        // Load the model
        Llama model;

        model = ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true);

        // Create a sampler
        Sampler sampler = selectSampler(
                model.configuration().vocabularySize,
                options.temperature(),
                options.topp(),
                options.seed()
        );

        // Choose execution path based on debug flags
        if (runDebugValidation) {
            runValidationComparison(model);
        } else if (runSingleForward) {
            runSingleForwardOperation(model, options);
        } else if (runComparison) {
            runComparisonInteractive(model, sampler, options);
        } else if (options.interactive()) {
            // Normal interactive mode
            runInteractive(model, sampler, options);
        } else {
            // Normal instruct mode
            runInstructOnce(model, sampler, options);
        }
    }

    // The original methods from LlamaApp for normal operation
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
            List<Integer> responseTokens = Llama.generateTokens(model, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
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

        Set<Integer> stopTokens = chatFormat.getStopTokens();
        List<Integer> responseTokens = Llama.generateTokens(model, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
            if (options.stream()) {
                if (!model.tokenizer().isSpecialToken(token)) {
                    System.out.print(model.tokenizer().decode(List.of(token)));
                }
            }
        });
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        if (!options.stream()) {
            String responseText = model.tokenizer().decode(responseTokens);
            System.out.println(responseText);
        }
    }
}