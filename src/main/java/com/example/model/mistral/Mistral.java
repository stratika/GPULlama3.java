package com.example.model.mistral;

import com.example.auxiliary.LastRunMetrics;
import com.example.auxiliary.format.MistralChatFormat;
import com.example.inference.InferenceEngine;
import com.example.inference.sampler.Sampler;
import com.example.model.Model;
import com.example.Options;
import com.example.loader.weights.ModelLoader;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.tokenizer.impl.MistralTokenizer;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tornadovm.TornadoVMMasterPlan;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

import static com.example.LlamaApp.USE_TORNADOVM;

/**
 * Llama class in mistral.java
 */
public record Mistral(MistralConfiguration configuration, Tokenizer tokenizer, Weights weights) implements Model {

    /* For explicit use */
    private MistralTokenizer getAsMistralTokenizer() { return (MistralTokenizer) tokenizer; }

    @Override
    public ModelLoader.ModelType getModelType() {
        return ModelLoader.ModelType.MISTRAL;
    }

    public State createNewState() {
        State state = new State(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get("<s>");
        return state;
    }

    public State createNewState(int batchsize) {
        State state = new State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<s>");
        return state;
    }

    @Override
    public void runInstructOnce(Sampler sampler, Options options) {
        State state = createNewState();
        MistralChatFormat chatFormat = new MistralChatFormat(getAsMistralTokenizer());
        TornadoVMMasterPlan tornadoVMPlan = null;

        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(chatFormat.getBeginOfText());

        if (options.suffix() != null) {
            promptTokens.addAll(chatFormat.encodeFillInTheMiddle(options.prompt(), options.suffix()));
        } else {
            promptTokens.addAll(chatFormat.encodeMessage(options.prompt(), true, true));
        }

          List<Integer> responseTokens;
        Set<Integer> stopTokens = chatFormat.getStopTokens();
        IntConsumer tokenConsumer = token -> {
            if (options.stream()) {
                int tokenType = getAsMistralTokenizer().getTokenType(token);
                if (tokenType == 1 || tokenType == 6) {
                    System.out.print(tokenizer.decode(List.of(token)));
                }
            }
        };

        if (USE_TORNADOVM) {
            tornadoVMPlan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, this);
            // Call generateTokensGPU without the token consumer parameter
            responseTokens = InferenceEngine.generateTokensGPU(this, state, 0, promptTokens, stopTokens,
                    options.maxTokens(), sampler, options.echo(), options.stream() ? tokenConsumer : null, tornadoVMPlan);
        } else {
            responseTokens = InferenceEngine.generateTokens(this, state, 0, promptTokens, stopTokens,
                    options.maxTokens(), sampler, options.echo(), tokenConsumer);
        }

        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        if (!options.stream()) {
            String responseText = tokenizer.decode(responseTokens);
            System.out.println(responseText);
        }

        LastRunMetrics.printMetrics();

        if (tornadoVMPlan != null) {
            tornadoVMPlan.freeTornadoExecutionPlan();
        }
    }
}
