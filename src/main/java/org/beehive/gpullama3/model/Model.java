package org.beehive.gpullama3.model;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.auxiliary.LastRunMetrics;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.IntConsumer;

import static org.beehive.gpullama3.LlamaApp.SHOW_PERF_INTERACTIVE;

public interface Model {

    Configuration configuration();

    Tokenizer tokenizer();

    Weights weights();

    ChatFormat chatFormat();

    TornadoVMMasterPlan tornadoVMPlan();

    void setTornadoVMPlan(TornadoVMMasterPlan plan);

    ModelType getModelType();

    State createNewState();

    State createNewState(int batchsize);

    /**
     * Wrapper for invoking the model-specific forward pass via InferenceCore.
     *
     * <p>
     * Delegates to the appropriate InferenceCore method based on the model type (e.g., {@code forwardJava}, {@code forwardJavaQwen3}).
     * </p>
     */
    void forward(State state, int token, int position);

    /**
     * Wrapper for invoking the model-specific {@code InferenceEngine.generateTokens} call.
     */
    List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo, IntConsumer onTokenGenerated);

    List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo, IntConsumer onTokenGenerated,
            TornadoVMMasterPlan tornadoVMPlan);

    /**
     * Model agnostic default implementation for interactive mode.
     *
     * @param sampler
     * @param options
     */
    default void runInteractive(Sampler sampler, Options options) {
        // Even though might be expensive, create state here for smoother interaction later
        State state = createNewState();
        List<Integer> conversationTokens = new ArrayList<>();
        ChatFormat chatFormat = chatFormat();
        TornadoVMMasterPlan tornadoVMPlan = null;

        if (!getModelType().equals(ModelType.QWEN_3) && !getModelType().equals(ModelType.PHI_3)) {
            conversationTokens.add(chatFormat.getBeginOfText());
        }

        if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }

        int startPosition = 0;
        Scanner in = new Scanner(System.in);

        // Initialize TornadoVM plan once at the beginning if GPU path is enabled
        if (Options.getDefaultOptions().useTornadovm() && tornadoVMPlan == null) {
            tornadoVMPlan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, this);
        }

        try {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                String userText = in.nextLine();
                if (List.of("quit", "exit").contains(userText)) {
                    break;
                }

                conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, userText)));
                conversationTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
                Set<Integer> stopTokens = chatFormat.getStopTokens();

                List<Integer> responseTokens;
                IntConsumer tokenConsumer = token -> {
                    if (options.stream()) {
                        if (tokenizer().shouldDisplayToken(token)) {
                            System.out.print(tokenizer().decode(List.of(token)));
                        }
                    }
                };

                // Choose between GPU and CPU path based on configuration
                if (Options.getDefaultOptions().useTornadovm()) {
                    // GPU path using TornadoVM
                    responseTokens = generateTokensGPU(state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler,
                            options.echo(), options.stream() ? tokenConsumer : null, tornadoVMPlan);
                } else {
                    // CPU path
                    responseTokens = generateTokens(state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler,
                            options.echo(), tokenConsumer);
                }

                // Include stop token in the prompt history, but not in the response displayed to the user.
                conversationTokens.addAll(responseTokens);
                startPosition = conversationTokens.size();
                Integer stopToken = null;
                if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                    stopToken = responseTokens.getLast();
                    responseTokens.removeLast();
                }
                if (!options.stream()) {
                    String responseText = tokenizer().decode(responseTokens);
                    System.out.println(responseText);
                }
                if (stopToken == null) {
                    System.err.println("\n Ran out of context length...\n Increase context length with by passing to llama-tornado --max-tokens XXX");
                    break;
                }
                System.out.print("\n");

                // Optionally print performance metrics after each response
                if (SHOW_PERF_INTERACTIVE) {
                    LastRunMetrics.printMetrics();
                }
            }
        } finally {
            // Clean up TornadoVM resources when exiting the chat loop
            if (Options.getDefaultOptions().useTornadovm() && tornadoVMPlan != null) {
                try {
                    tornadoVMPlan.freeTornadoExecutionPlan();
                } catch (Exception e) {
                    System.err.println("Error while cleaning up TornadoVM resources: " + e.getMessage());
                }
            }
        }
    }

    /**
     * Model agnostic default implementation for instruct mode.
     *
     * @param sampler
     * @param options
     */
    default String runInstructOnce(Sampler sampler, Options options) {
        State state = createNewState();
        ChatFormat chatFormat = chatFormat();
        TornadoVMMasterPlan tornadoVMPlan = null;

        List<Integer> promptTokens = new ArrayList<>();

        if (!getModelType().equals(ModelType.QWEN_3) && !getModelType().equals(ModelType.PHI_3)) {
            promptTokens.add(chatFormat.getBeginOfText());
        }

        if (options.systemPrompt() != null) {
            promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }

        // Initialize TornadoVM plan once at the beginning if GPU path is enabled
        if (Options.getDefaultOptions().useTornadovm() && tornadoVMPlan == null) {
            tornadoVMPlan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, this);
        }

        promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        List<Integer> responseTokens;

        IntConsumer tokenConsumer = token -> {
            if (options.stream()) {
                if (tokenizer().shouldDisplayToken(token)) {
                    System.out.print(tokenizer().decode(List.of(token)));
                }
            }
        };

        Set<Integer> stopTokens = chatFormat.getStopTokens();

        if (Options.getDefaultOptions().useTornadovm()) {
            // GPU path using TornadoVM - Call generateTokensGPU without the token consumer parameter
            responseTokens = generateTokensGPU(state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), options.stream() ? tokenConsumer : null, tornadoVMPlan);
        } else {
            // CPU path
            responseTokens = generateTokens(state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), tokenConsumer);
        }

        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }

        String responseText = "";
        if (!options.stream()) {
            responseText = tokenizer().decode(responseTokens);
        }

        if (tornadoVMPlan != null) {
            tornadoVMPlan.freeTornadoExecutionPlan();
        }

        return responseText;
    }

    /**
     * Model agnostic default implementation for instruct mode.
     *
     * @param sampler
     * @param options
     */
    default String runInstructOnceLangChain4J(Sampler sampler, Options options, Consumer<String> tokenCallback) {
        State state = createNewState();
        ChatFormat chatFormat = chatFormat();
        TornadoVMMasterPlan tornadoVMPlan = null;

        List<Integer> promptTokens = new ArrayList<>();

        if (!getModelType().equals(ModelType.QWEN_3) && !getModelType().equals(ModelType.PHI_3)) {
            promptTokens.add(chatFormat.getBeginOfText());
        }

        if (options.systemPrompt() != null) {
            promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }

        // Initialize TornadoVM plan once at the beginning if GPU path is enabled
        if (Options.getDefaultOptions().useTornadovm() && tornadoVMPlan == null) {
            tornadoVMPlan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, this);
        }

        promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        List<Integer> responseTokens;

        IntConsumer tokenConsumer = token -> {
            if (tokenizer().shouldDisplayToken(token)) {
                String piece = tokenizer().decode(List.of(token));
                if (options.stream() && tokenCallback != null) {
                    tokenCallback.accept(piece);  // ✅ send to LangChain4j handler
                }
            }
        };

        Set<Integer> stopTokens = chatFormat.getStopTokens();

        if (Options.getDefaultOptions().useTornadovm()) {
            // GPU path using TornadoVM Call generateTokensGPU without the token consumer parameter
            responseTokens = generateTokensGPU(state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), options.stream() ? tokenConsumer : null, tornadoVMPlan);
        } else {
            // CPU path
            responseTokens = generateTokens(state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), tokenConsumer);
        }

        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }

        String responseText = tokenizer().decode(responseTokens);

        if (!options.stream()) {
            responseText = tokenizer().decode(responseTokens);
        }

        if (tornadoVMPlan != null) {
            tornadoVMPlan.freeTornadoExecutionPlan();
        }

        return responseText;
    }
}
