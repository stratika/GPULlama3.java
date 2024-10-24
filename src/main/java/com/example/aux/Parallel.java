package com.example.aux;

import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public final class Parallel {
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }
}