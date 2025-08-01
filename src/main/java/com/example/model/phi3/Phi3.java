package com.example.model.phi3;

import com.example.inference.sampler.Sampler;
import com.example.inference.state.State;
import com.example.inference.weights.Weights;
import com.example.model.AbstractModel;
import com.example.model.Configuration;
import com.example.model.ModelType;
import com.example.model.format.ChatFormat;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public class Phi3 extends AbstractModel {
    protected Phi3(Tokenizer tokenizer, Weights weights, ChatFormat chatFormat, TornadoVMMasterPlan plan) {
        super(tokenizer, weights, chatFormat, plan);
    }

    @Override
    public Configuration configuration() {
        return null;
    }

    @Override
    public Tokenizer tokenizer() {
        return null;
    }

    @Override
    public ModelType getModelType() {
        return null;
    }

    @Override
    public State createNewState() {
        return null;
    }

    @Override
    public State createNewState(int batchsize) {
        return null;
    }

    @Override
    public void forward(State state, int token, int position) {

    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        return List.of();
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        return List.of();
    }
}
