package com.example.core.model;

import com.example.core.types.Float16;

public enum GGMLType {
    // Floating point types
//    F16(Float16.BYTES),                  // 16-bit floating point
//    BF16(GGMLType.BFLOAT16_BYTES),      // bfloat16 format (used for ML)
//    F32(Float.BYTES),                   // 32-bit floating point
//    F64(Double.BYTES),                 // 64-bit floating point
//
//    // Integer types
//    I8(Byte.BYTES),                    // 8-bit signed integer
//    I16(Short.BYTES),                  // 16-bit signed integer
//    I32(Integer.BYTES),                // 32-bit signed integer
//    I64(Long.BYTES),                   // 64-bit signed integer
//
//    // Legacy quantizations (4-bit and 5-bit quantizations)
//    Q4_0(Float16.BYTES + 16 * Byte.BYTES, 32),                    // 4-bit quantization with scale
//    Q4_1(2 * Float16.BYTES + 16 * Byte.BYTES, 32),                // 4-bit quantization with scale and offset
//    Q5_0(Integer.MAX_VALUE),                                      // Unsupported or placeholder
//    Q5_1(Integer.MAX_VALUE),                                      // Unsupported or placeholder
//    Q8_0(Float16.BYTES + 32 * Byte.BYTES, 32),                    // 8-bit quantization with scale
//    Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, 32),                  // 8-bit quantization with row-wise scale and offset
//
//    // Removed quantization types
//    UNSUPPORTED_Q4_2(Integer.MAX_VALUE), // Removed support
//    UNSUPPORTED_Q4_3(Integer.MAX_VALUE), // Removed support
//
//    // K-quantizations: optimized for block quantization (e.g., GPTQ)
//    Q2_K(Integer.MAX_VALUE),                                            // Placeholder or unsupported 2-bit K quant
//    Q3_K(Integer.MAX_VALUE),                                            // Placeholder or unsupported 3-bit K quant
//    Q4_K(2 * Float16.BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2, GGMLType.QK_K), // 4-bit block quantization
//    Q5_K(2 * Float16.BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 8 + GGMLType.QK_K / 2, GGMLType.QK_K), // 5-bit block quant
//    Q6_K(GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + Float16.BYTES, GGMLType.QK_K), // 6-bit block quant
//    Q8_K(Integer.MAX_VALUE),                                            // Placeholder for 8-bit K quant (possibly unused)
//
//    // Extended/experimental IQ quantizations
//    IQ1_M(Integer.MAX_VALUE),
//    IQ1_S(Integer.MAX_VALUE),
//    IQ2_XXS(Integer.MAX_VALUE),
//    IQ2_XS(Integer.MAX_VALUE),
//    IQ2_S(Integer.MAX_VALUE),
//    IQ3_XXS(Integer.MAX_VALUE),
//    IQ3_S(Integer.MAX_VALUE),
//    IQ4_XS(Integer.MAX_VALUE),
//    IQ4_NL(Integer.MAX_VALUE),
//
//    // Ternary quantizations (possibly ternary quant bits)
//    TQ1_0(Integer.MAX_VALUE),
//    TQ2_0(Integer.MAX_VALUE),
//
//    // Custom or hybrid quant formats (naming convention suggests mixed 4-bit blocks)
//    Q4_0_4_4(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
//    Q4_0_4_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
//    Q4_0_8_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32);
//
//    F32(Float.BYTES),
//    F16(Float16.BYTES),
//    Q4_0(Float16.BYTES + 16 * Byte.BYTES, 32),
//    Q4_1(2 * Float16.BYTES + 16 * Byte.BYTES, 32),
//    UNSUPPORTED_Q4_2(Integer.MAX_VALUE), // support has been removed
//    UNSUPPORTED_Q4_3(Integer.MAX_VALUE), // support has been removed
//    Q5_0(Integer.MAX_VALUE),
//    Q5_1(Integer.MAX_VALUE),
//    Q8_0(Float16.BYTES + 32 * Byte.BYTES, 32),
//    Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, 32),
//    // k-quantizations
//    Q2_K(Integer.MAX_VALUE),
//    Q3_K(Integer.MAX_VALUE),
//    Q4_K(2 * Float16.BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2, GGMLType.QK_K),
//    Q5_K(2 * Float16.BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 8 + GGMLType.QK_K / 2, GGMLType.QK_K),
//    Q6_K(GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + Float16.BYTES, GGMLType.QK_K),
//    Q8_K(Integer.MAX_VALUE),
//    I8(Byte.BYTES),
//    I16(Short.BYTES),
//    I32(Integer.BYTES);

    F32(Float.BYTES),
    F16(GGMLType.FLOAT16_BYTES),
    Q4_0(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    Q4_1(2 * GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    UNSUPPORTED_Q4_2(Integer.MAX_VALUE), // support has been removed
    UNSUPPORTED_Q4_3(Integer.MAX_VALUE), // support has been removed
    Q5_0(Integer.MAX_VALUE),
    Q5_1(Integer.MAX_VALUE),
    Q8_0(GGMLType.FLOAT16_BYTES + 32 * Byte.BYTES, 32),
    Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, 32),
    // k-quantizations
    Q2_K(Integer.MAX_VALUE),
    Q3_K(Integer.MAX_VALUE),
    Q4_K(2 * GGMLType.FLOAT16_BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2, GGMLType.QK_K),
    Q5_K(2 * GGMLType.FLOAT16_BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 8 + GGMLType.QK_K / 2, GGMLType.QK_K),
    Q6_K(GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + GGMLType.FLOAT16_BYTES, GGMLType.QK_K),
    Q8_K(Integer.MAX_VALUE),

    IQ2_XXS(Integer.MAX_VALUE),
    IQ2_XS(Integer.MAX_VALUE),
    IQ3_XXS(Integer.MAX_VALUE),
    IQ1_S(Integer.MAX_VALUE),
    IQ4_NL(Integer.MAX_VALUE),
    IQ3_S(Integer.MAX_VALUE),
    IQ2_S(Integer.MAX_VALUE),
    IQ4_XS(Integer.MAX_VALUE),

    I8(Byte.BYTES),
    I16(Short.BYTES),
    I32(Integer.BYTES),
    I64(Long.BYTES),
    F64(Double.BYTES),
    IQ1_M(Integer.MAX_VALUE),
    BF16(GGMLType.BFLOAT16_BYTES),
    Q4_0_4_4(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    Q4_0_4_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    Q4_0_8_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    TQ1_0(Integer.MAX_VALUE),
    TQ2_0(Integer.MAX_VALUE);

    public static final int BFLOAT16_BYTES = 2;
    public static final int FLOAT16_BYTES = 2;

    private static final GGMLType[] VALUES = values();

    private final int typeSize;

    private final int blockSize;

    public int getTypeSize() {
        return typeSize;
    }

    public int getBlockSize() {
        return blockSize;
    }

    public static GGMLType fromId(int id) {
        return VALUES[id];
    }

    GGMLType(int typeSize) {
        this(typeSize, 1);
    }

    public long byteSizeFor(int numberOfElements) {
        long t = numberOfElements * (long) getTypeSize();
        assert t % getBlockSize() == 0;
        return Math.toIntExact(t / getBlockSize());
    }

    public static final int QK_K = 256; // or 64?

    GGMLType(int typeSize, int blockSize) {
        assert blockSize > 0;
        assert typeSize > 0;
        assert isPowerOf2(blockSize);
        this.typeSize = typeSize;
        this.blockSize = blockSize;
    }

    private static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}