package com.example.model.mistral;

import com.example.model.Model;
import com.example.loader.weights.ModelLoader;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.tokenizer.impl.MistralTokenizer;
import com.example.tokenizer.impl.Tokenizer;

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

}
