package org.beehive.gpullama3.core.model.tensor;

import org.beehive.gpullama3.core.model.GGMLType;

import java.lang.foreign.MemorySegment;

public record GGMLTensorEntry(MemorySegment mappedFile, String name, GGMLType ggmlType, int[] shape,
                              MemorySegment memorySegment) {
}
