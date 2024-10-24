package com.example.inference;

import com.example.core.model.tensor.FloatTensor;

@FunctionalInterface
public interface Sampler {
    int sampleToken(FloatTensor logits);

    Sampler ARGMAX = FloatTensor::argmax;
}
