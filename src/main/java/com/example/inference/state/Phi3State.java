package com.example.inference.state;

import com.example.model.Configuration;

public class Phi3State extends  State{
    /**
     * last index in previous block
     *
     * @param config
     * @param batchsize
     */
    public Phi3State(Configuration config, int batchsize) {
        super(config, batchsize);
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        return null;
    }
}
