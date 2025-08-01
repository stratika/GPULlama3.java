package com.example.tokenizer.impl;

import com.example.tokenizer.vocabulary.Vocabulary;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Phi3Tokenizer implements Tokenizer {

    private static final String SPM_UNDERSCORE = "\u2581";
    private static final String PHI3_PATTERN = "\\S+|\\s+"; // Define appropriate pattern for Phi3
    private final Vocabulary vocabulary;
    // general fields
    private final Pattern compiledPattern;
    // model-specific fields
    private final Map<String, Integer> specialTokens;
    private final int[] tokenType;
    private final int byte0;

    public Phi3Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        // load from metadata
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        List<Integer> specialTokensList = IntStream.range(0, vocabulary.size()).filter(t -> tokenTypes[t] != 1 && tokenTypes[t] != 6).boxed().toList();
        Map<String, Integer> specialTokens = IntStream.range(0, specialTokensList.size()).boxed().collect(Collectors.toMap(t -> vocabulary.get(specialTokensList.get(t)), specialTokensList::get));

        // init tokenizer object fields
        this.vocabulary = vocabulary;
        this.compiledPattern = Pattern.compile(PHI3_PATTERN);
        this.specialTokens = new HashMap<>(specialTokens);
        this.tokenType = tokenTypes;
        this.byte0 = vocabulary.getIndex("<0x00>").orElseThrow();
    }

    @Override
    public String regexPattern() {
        return "";
    }

    @Override
    public Map<String, Integer> getSpecialTokens() {
        return Map.of();
    }

    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return false;
    }

    @Override
    public boolean shouldDisplayToken(int token) {
        return false;
    }

    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        return List.of();
    }

    @Override
    public List<Integer> encodeAsList(String pText) {
        String text = pText.replace(" ", SPM_UNDERSCORE);
        text = pText.startsWith(SPM_UNDERSCORE) ? text : SPM_UNDERSCORE + text;
        final int textLen = text.length();

        final List<Integer> tokens = new ArrayList<>();
        final int vocSize = vocabulary.size();
        int offset = 0;
        while (offset < textLen) {
            String curVoc = null;
            int token = -1;
            for (int j = 0; j < vocSize; j++) {
                final String voc = vocabulary.get(j);
                if (text.startsWith(voc, offset) && (curVoc == null || curVoc.length() < voc.length())) {
                    curVoc = voc;
                    token = j;
                }
            }
            if (curVoc == null) {
                // Try <0xE7>... of character or surrogate (emoji).
                final int len = (offset + 1 < textLen) && Character.isHighSurrogate(text.charAt(offset)) ? 2 : 1;
                final byte[] bufUtf8 = text.substring(offset, offset + len).getBytes(StandardCharsets.UTF_8);
                for (int i = 0; i < bufUtf8.length; i++) {
                    final String sHex = String.format("<0x%02x>", bufUtf8[i] & 0xff);
                    token = -1;
                    for (int j = 0; j < vocSize; j++) {
                        if (sHex.equalsIgnoreCase(vocabulary.get(j))) {
                            token = j;
                        }
                    }
                    if (token == -1) {
                        throw new RuntimeException(String.format("Can't tokenize text at offset %d (%c / (%d, sHex %s)), tokens = %s, text: %s", offset, text.charAt(offset), i, sHex, tokens, text));
                    }
                    tokens.add(token);
                }
                offset += len;
                continue;
            }
            tokens.add(token);
            offset += curVoc.length();
        }
        return tokens;
    }

    @Override
    public String decode(List<Integer> tokens) {
        final StringBuilder sb = new StringBuilder();
        for (Integer token : tokens) {
            sb.append(vocabulary.get(token));
        }
        return sb.toString().replace(SPM_UNDERSCORE, " ");
    }
}
