package com.example.inference.engine.impl;

public  final class Configuration {
    /** Transformer embedding dimension */
    public final int dim;

    /** Hidden dimension size for feed-forward network layers */
    public final int hiddenDim;

    /** Number of transformer layers in the model */
    public final int numberOfLayers;

    /** Number of attention heads for queries */
    public final int numberOfHeads;

    /** Number of key/value heads (can be fewer than query heads in multi-query attention) */
    public final int numberOfKeyValueHeads;

    /** Size of the vocabulary (token set) */
    public final int vocabularySize;

    /** Maximum sequence length the model can process */
    public final int contextLength;

    /** Epsilon value for RMSNorm layers (stabilizes normalization) */
    public final float rmsNormEps;

    /** Base value for RoPE (Rotary Position Embedding) calculations */
    public final float ropeTheta;

    /** Size of each attention head (derived from dim / numberOfHeads) */
    public final int headSize;

    /** Key/value dimension (derived from dim * numberOfKeyValueHeads / numberOfHeads) */
    public final int kvDim;

    /** Multiplier for key/value sharing in multi-query attention */
    public final int kvMul;

    /**

    /**
     * Constructs a new Configuration with the specified parameters.
     *
     * @param dim Transformer embedding dimension
     * @param hiddenDim Hidden dimension for feed-forward layers
     * @param numberOfLayers Number of transformer layers
     * @param numberOfHeads Number of attention heads
     * @param numberOfKeyValueHeads Number of key/value heads
     * @param vocabularySize Size of the vocabulary
     * @param contextLength Maximum sequence length
     * @param rmsNormEps Epsilon for RMSNorm
     * @param ropeTheta Base value for RoPE calculations
     */
    public Configuration(int dim, int hiddenDim, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads, int vocabularySize, int contextLength, float rmsNormEps, float ropeTheta) {
        this.dim = dim;
        this.hiddenDim = hiddenDim;
        this.numberOfLayers = numberOfLayers;
        this.numberOfHeads = numberOfHeads;
        this.numberOfKeyValueHeads = numberOfKeyValueHeads;
        this.vocabularySize = vocabularySize;
        this.contextLength = contextLength;
        this.rmsNormEps = rmsNormEps;
        this.ropeTheta = ropeTheta;
        this.headSize = dim / numberOfHeads;
        this.kvDim =  dim * numberOfKeyValueHeads / numberOfHeads;
        this.kvMul = numberOfHeads / numberOfKeyValueHeads;
    }

    /**
     * Creates a new Configuration with a different context length.
     *
     * @param newContextLength The new context length to use
     * @return A new Configuration instance with updated context length,
     *         or the current instance if newContextLength is negative
     */
    public Configuration withContextLength(int newContextLength) {
        if (newContextLength < 0) {
            return this; // no change
        }
        return new Configuration(
                this.dim,
                this.hiddenDim,
                this.numberOfLayers,
                this.numberOfHeads,
                this.numberOfKeyValueHeads,
                this.vocabularySize,
                newContextLength,
                this.rmsNormEps,
                this.ropeTheta
        );
    }
}

