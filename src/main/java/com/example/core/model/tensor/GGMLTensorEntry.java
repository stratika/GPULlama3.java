package com.example.core.model.tensor;

import com.example.core.model.GGMLType;

import java.lang.foreign.MemorySegment;

public record GGMLTensorEntry(MemorySegment mappedFile, String name, GGMLType ggmlType, int[] shape,
                              MemorySegment memorySegment) {
}
