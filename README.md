# TornadoVM Llama3 Java Integration

Integration of **Llama3 models** with **TornadoVM** to enable accelerated inference on Java using GPUs and CPUs. This project allows you to run Llama3 inference efficiently, leveraging TornadoVM's parallel computing features for enhanced performance.


This project builds on [Llama3.java](https://github.com/mukel/llama3.java), based on the original [Llama 3](https://github.com/meta-llama/llama3), [3.1](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1), and [3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) models, with TornadoVM support for parallelism and hardware acceleration.

Previous intergration of TornadoVM and Llama2 it can be found in [llama2.tornadovm](https://github.com/mikepapadim/llama2.tornadovm.java).

---

## Features

- TornadoVM-accelerated Llama 3 inference with Java
- Supports GGUF format models
- Instruction-following and chat modes
- Optimized for TornadoVM’s GPU and CPU modes
- Interactive CLI with `--chat` and `--instruct` options

## Setup

### Prerequisites

Ensure you have the following installed and configured:

- **Java 21+**: Required for Vector API support.
- **TornadoVM**: To install **TornadoVM**, you'll need to set up the environment variables `TORNADO_ROOT` and `TORNADO_SDK` as part of the configuration process.
  For detailed installation instructions, visit the [TornadoVM GitHub repository](https://github.com/beehive-lab/TornadoVM).
- **Maven**: For building the Java project.

### Download Model Files

Download quantized `.gguf` files for Llama3 models from [Hugging Face](https://huggingface.co/mukel/), or using the following `curl` commands:

```bash
# Llama 3.2 (1B) -> Q4_0
curl -L -O https://huggingface.co/mukel/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf

# Llama 3.2 (1B) -> Q8_0
curl -L -O https://huggingface.co/mukel/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf

# Llama 3.2 (3B) -> Q4_0
curl -L -O https://huggingface.co/mukel/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf

# Llama 3.1 (8B) -> Q8_0
curl -L -O https://huggingface.co/mukel/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_0.gguf
```

### Download the Tokenizer.bin
```bash
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```

### Configuration

Set up environment variables by editing and sourcing the `set_paths` script:

```bash
export TORNADO_ROOT=/path/to/TornadoVM
export TORNADO_SDK=${TORNADO_ROOT}/bin/sdk
export LLAMA_ROOT=/path/to/this-project
export PATH="${PATH}:${LLAMA_ROOT}/bin"
```

### Building the Project

```bash
mvn clean install
```


## Running Llama3 Models

The `tornado-llama` script executes Llama3 models on TornadoVM. By default, models run on the CPU; specify `--gpu` to enable GPU acceleration.

### Usage Examples

#### Basic Inference
Run a model with a text prompt:

```bash
./tornado-llama --model Llama-3.2-1B-Instruct-Q4_0.gguf --prompt "Explain the benefits of GPU acceleration."
```

#### GPU Execution
Enable GPU acceleration by adding the `--gpu` flag:

```bash
./tornado-llama --gpu --model Llama-3.2-1B-Instruct-Q4_0.gguf --prompt "What is TornadoVM?"
```

#### Interactive Chat Mode
Start an interactive session:

```bash
./tornado-llama --model Llama-3.2-1B-Instruct-Q4_0.gguf --interactive
```

#### Instruction-following with Streaming
Run the model in instruction-following mode with token streaming:

```bash
./tornado-llama --model Llama-3.2-1B-Instruct-Q4_0.gguf --prompt "List Java advantages." --instruct --stream true
```

## Command Line Options

Supported command-line options include:

- `--model` - Specifies the model file path (e.g., `Llama-3.2-1B-Instruct-Q4_0.gguf`)
- `--prompt` - Input text prompt
- `--gpu` - Enables GPU mode for faster execution
- `--temperature <float>` - Sampling temperature to control diversity
- `--top-p <float>` - Probability threshold for sampling
- `--max-tokens <int>` - Limits the number of generated tokens
- `--stream <true/false>` - Streams output tokens
- `--interactive` - Enables continuous input mode
- `--instruct` - Instruction-following model

## License

MIT
