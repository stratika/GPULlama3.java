package com.example.model;

import com.example.inference.weights.Weights;
import com.example.model.format.ChatFormat;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tornadovm.TornadoVMMasterPlan;

public abstract class AbstractModel implements Model {

    protected Tokenizer tokenizer;
    protected Weights weights;
    protected ChatFormat chatFormat;
    /**
     * Represents the master plan for the TornadoVM execution in the context of the model.
     * This variable is used to manage the execution flow or strategy within the TornadoVM environment.
     * <p>
     * Initialized *only* when Tornado is used, via {@link TornadoVMMasterPlan#initializeTornadoVMPlan}.
     * </p>
     */
    protected TornadoVMMasterPlan plan;

    protected AbstractModel(Tokenizer tokenizer, Weights weights, ChatFormat chatFormat, TornadoVMMasterPlan plan) {
        this.tokenizer = tokenizer;
        this.weights = weights;
        this.chatFormat = chatFormat;
        this.plan = plan;
    }

    // Common methods across models

    public Weights weights() {
        return weights;
    }

    public ChatFormat chatFormat() {
        return chatFormat;
    }

    public TornadoVMMasterPlan tornadoVMPlan() {
        return plan;
    }

    public void setTornadoVMPlan(TornadoVMMasterPlan plan) {
        this.plan = plan;
    }

}
