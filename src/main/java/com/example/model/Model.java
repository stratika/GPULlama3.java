package com.example.model;

import com.example.inference.sampler.Sampler;
import com.example.Options;
import com.example.loader.weights.ModelLoader.ModelType;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public interface Model {
    Configuration configuration();
    Tokenizer tokenizer();
    Weights weights();

    ModelType getModelType();

    State createNewState();
    State createNewState(int batchsize);

    void runInteractive(Sampler sampler, Options options);
    void runInstructOnce(Sampler sampler, Options options);
}
