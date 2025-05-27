package com.example.auxiliary.format;

import com.example.tokenizer.impl.LlamaTokenizer;
import com.example.tokenizer.impl.Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class LlamaChatFormat implements ChatFormat {

    final LlamaTokenizer tokenizer;
    public final int beginOfText;
    final int endHeader;
    final int startHeader;
    final int endOfTurn;
    final int endOfText;
    final int endOfMessage;
    final Set<Integer> stopTokens;

    public LlamaChatFormat(LlamaTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.get("<|begin_of_text|>");
        this.startHeader = specialTokens.get("<|start_header_id|>");
        this.endHeader = specialTokens.get("<|end_header_id|>");
        this.endOfTurn = specialTokens.get("<|eot_id|>");
        this.endOfText = specialTokens.get("<|end_of_text|>");
        this.endOfMessage = specialTokens.getOrDefault("<|eom_id|>", -1); // only in 3.1
        this.stopTokens = Set.of(endOfText, endOfTurn);
    }

    public Tokenizer getTokenizer() {
        return tokenizer;
    }

    public Set<Integer> getStopTokens() {
        return stopTokens;
    }

    public List<Integer> encodeHeader(LlamaChatFormat.Message message) {
        List<Integer> tokens = new ArrayList<>();
        LlamaTokenizer llamaTokenizer = (LlamaTokenizer) this.tokenizer;
        tokens.add(startHeader);
        tokens.addAll(llamaTokenizer.encodeAsList(message.role().name()));
        tokens.add(endHeader);
        tokens.addAll(llamaTokenizer.encodeAsList("\n"));
        return tokens;
    }

    public List<Integer> encodeMessage(LlamaChatFormat.Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfTurn);
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<LlamaChatFormat.Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfText);
        for (LlamaChatFormat.Message message : dialog) {
            tokens.addAll(this.encodeMessage(message));
        }
        if (appendAssistantTurn) {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(this.encodeHeader(new LlamaChatFormat.Message(LlamaChatFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }

    public record Message(LlamaChatFormat.Role role, String content) {
    }

    public record Role(String name) {
        public static LlamaChatFormat.Role SYSTEM = new LlamaChatFormat.Role("system");
        public static LlamaChatFormat.Role USER = new LlamaChatFormat.Role("user");
        public static LlamaChatFormat.Role ASSISTANT = new LlamaChatFormat.Role("assistant");

        @Override
        public String toString() {
            return name;
        }
    }
}