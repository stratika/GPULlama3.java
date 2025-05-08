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
import com.example.tornadovm.TornadoVMCompute;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.IOException;
import java.util.*;
import java.util.function.IntConsumer;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;


public class LlamaApp {
    public static final boolean USE_VECTOR_API = Boolean.parseBoolean(System.getProperty("llama.VectorAPI", "true"));
    public static final boolean USE_AOT = Boolean.parseBoolean(System.getProperty("llama.AOT", "false"));
    public static final boolean TORNADOVM = Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"));


    static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        Sampler sampler;
        if (temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            sampler = Sampler.TENSOR_ARGMAX; // Use TENSOR_ARGMAX instead of ARGMAX
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
                // Handle both FloatTensor and FloatArray types
                if (logits instanceof FloatTensor) {
                    // For FloatTensor, use its methods directly
                    FloatTensor tensorLogits = (FloatTensor) logits;
                    tensorLogits.divideInPlace(0, tensorLogits.size(), temperature);
                    tensorLogits.softmaxInPlace(0, tensorLogits.size());
                } else if (logits instanceof FloatArray) {
                    // For FloatArray, use our helper class
                    FloatArray arrayLogits = (FloatArray) logits;
                    FloatArrayUtils.divideInPlace(arrayLogits, 0, arrayLogits.getSize(), temperature);
                    FloatArrayUtils.softmaxInPlace(arrayLogits, 0, arrayLogits.getSize());
                } else {
                    throw new IllegalArgumentException("Unsupported logits type: " +
                            (logits != null ? logits.getClass().getName() : "null"));
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
        if (TornadoVMCompute.TORNADOVM) {
            // Call generateTokensGPU without the token consumer parameter
            responseTokens = Llama.generateTokensGPU(model, state, 0, promptTokens, stopTokens,
                    options.maxTokens(), sampler, options.echo());

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
            responseTokens = Llama.generateTokens(model, state, 0, promptTokens, stopTokens,
                    options.maxTokens(), sampler, options.echo(), tokenConsumer);
        }

        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        if (!options.stream()) {
            String responseText = model.tokenizer().decode(responseTokens);
            System.out.println(responseText);
        }
    }

    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Llama model;
        if (USE_AOT) {
             model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
        } else  {
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



