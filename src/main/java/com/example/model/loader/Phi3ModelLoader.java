package com.example.model.loader;

import com.example.core.model.GGUF;
import com.example.model.Model;

import java.nio.channels.FileChannel;

public class Phi3ModelLoader extends ModelLoader {
    public Phi3ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
        super(fileChannel, gguf, contextLength, loadWeights);
    }

    @Override
    public Model loadModel() {
        return null;
    }
}
