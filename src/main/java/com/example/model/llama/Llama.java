package com.example.model.llama;

import com.example.model.Model;
import com.example.loader.weights.ModelLoader;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.tokenizer.impl.LlamaTokenizer;
import com.example.tokenizer.impl.Tokenizer;

public record Llama(LlamaConfiguration configuration, Tokenizer tokenizer, Weights weights) implements Model {
    private static final int BATCH_SIZE = Integer.getInteger("llama.BatchSize", 16);

    /* For explicit use */
    private LlamaTokenizer getAsLlamaTokenizer() { return (LlamaTokenizer) tokenizer; }

    @Override
    public ModelLoader.ModelType getModelType() {
        return ModelLoader.ModelType.LLAMA_3;
    }

    @Override
    public State createNewState() {
        State state = new State(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

}

