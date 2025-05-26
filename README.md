# GPULlama3.java powered by TornadoVM
![Java Version](https://img.shields.io/badge/java-21+-blue?style=for-the-badge&logo=openjdk)
![TornadoVM](https://img.shields.io/badge/TornadoVM-enabled-green?style=for-the-badge&logo=apache)
![OpenCL](https://img.shields.io/badge/OpenCL-supported-blue?style=for-the-badge&logo=khronos)
![CUDA](https://img.shields.io/badge/CUDA/PTX-supported-76B900?style=for-the-badge&logo=nvidia)
[![GPULlama3.java DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/beehive-lab/GPULlama3.java)

[//]: # (![Last Commit]&#40;https://img.shields.io/github/last-commit/beehive-lab/GPULlama3.java?style=for-the-badge&#41;)

[//]: # (![Stars]&#40;https://img.shields.io/github/stars/beehive-lab//GPULlama3.java?style=social&#41;)

[//]: # (![Forks]&#40;https://img.shields.io/github/forks/beehive-lab//GPULlama3.java?style=social&#41;)

-----------
<table style="border: none;">
<tr style="border: none;">
<td style="width: 40%; vertical-align: middle; border: none;">
<img src="docs/java-tornado-gpu.jpg" width="100%">
</td>
<td style="vertical-align: middle; padding-left: 20px; border: none;">
Integration of <strong>Llama3 models</strong> with <strong>TornadoVM</strong> to enable accelerated inference on Java using GPUs and CPUs. This project allows you to run Llama3 inference efficiently, leveraging TornadoVM's parallel computing features for enhanced performance.
<br><br>
This project builds on <a href="https://github.com/mukel/llama3.java">Llama3.java</a>, based on the original <a href="https://github.com/meta-llama/llama3">Llama 3</a>, <a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1">3.1</a>, and <a href="https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/">3.2</a> models, with TornadoVM support for parallelism and hardware acceleration.
<br><br>
Thanks to <a href="https://github.com/mukel">Alfonso² Peterssen</a> for the original implementation of Llama3.java.
<br><br>
Previous intergration of TornadoVM and Llama2 it can be found in <a href="https://github.com/mikepapadim/llama2.tornadovm.java">llama2.tornadovm</a>.
</td>
</tr>
</table>

-----------

### TornadoVM-Accelerated Inference Performance and Optimization Status

This table shows inference performance across different hardware and quantization options.

| Vendor / Backend             | Hardware     | Llama-3.2-1B-Instruct | Llama-3.2-1B-Instruct | Llama-3.2-3B-Instruct | Optimizations |
|:----------------------------:|:------------:|:---------------------:|:---------------------:|:---------------------:|:-------------:|
|                              |              | **Q8_0**              | **Q4_0**              | **Q4_0**              |  **Support**  |
| **NVIDIA / OpenCL-PTX**      | RTX 3070     | 52 tokens/s         | 50.56 tokens/s         | 22.96 tokens/s         |       ✅      |
|                              | RTX 4090     | XXXX tokens/s         | XXXX tokens/s         | XXXX tokens/s         |       ✅      |
|                              | RTX 5090     | 47.68 tokens/s         | 94.71 tokens/s         | 96.65 tokens/s         |       ✅      |
|                              | H100         | XXXX tokens/s         | XXXX tokens/s         | XXXX tokens/s         |       ✅      |
| **Intel / OpenCL**           | Arc A770     | 15.65 tokens/s         | 15.09 tokens/s         | 7.02 tokens/s         |       ✅      |
| **Apple Silicon / OpenCL**   | M3 Pro       | 14.04 tokens/s         | 13.83 tokens/s         | 6.78 tokens/s         |      (WIP)    |
|                              | M4 Pro       | 16.77 tokens/s         | 16.67 tokens/s         | 8.56 tokens/s         |      (WIP)    |
| **AMD / OpenCL**             | Radeon RX    | (WIP)                 | (WIP)                 | (WIP)                 |      (WIP)    |

-----------

## Setup & Configuration

### Prerequisites

Ensure you have the following installed and configured:

- **Java 21**: Required for Vector API support & TornadoVM.
- [TornadoVM](https://github.com/beehive-lab/TornadoVM) with OpenCL or PTX backends.
- [Maven](https://maven.apache.org/): For building the Java project.

### Install, Build, and Run

When cloning this repository, use the `--recursive` flag to ensure that TornadoVM is properly included as submodule:

```bash
# Clone the repository with all submodules
git clone --recursive https://github.com/beehive-lab/GPULlama3.java.git

# Update the submodules to match the exact commit point recorded in this repository
git submodule update --recursive

# Navigate to the project directory
cd GPULlama3.java

# Enter the TornadoVM submodule directory
cd external/tornadovm

# Optional: Create and activate a Python virtual environment if needed
python3 -m venv venv
source ./venv/bin/activate

# Install TornadoVM with a supported JDK 21 and select the backends (--backend opencl,ptx).
# To see the compatible JDKs run: ./bin/tornadovm-installer --listJDKs
# For example, to install with OpenJDK 21 and build the OpenCL backend, run: 
./bin/tornadovm-installer --jdk jdk21 --backend opencl

# Source the TornadoVM environment variables
source setvars.sh

# Navigate back to the project root directory
cd ../../

# Make the llama-tornado script executable
chmod +x llama-tornado

# Source the project-specific environment paths -> this will ensure the correct paths are set for the project and the TornadoVM SDK
# Expect to see: [INFO] Environment configured for Llama3 with TornadoVM at: /home/YOUR_PATH_TO_TORNADOVM
source set_paths

# Build the project using Maven (skip tests for faster build)
# mvn clean package -DskipTests or just make
make

# Run the model (make sure you have downloaded the model file first -  see below)
./llama-tornado --gpu  --verbose-init --opencl --model Llama-3.2-1B-Instruct-Q4_0.gguf --prompt "tell me a joke"
```
-----------

The above model can we swapped with one of the other models, such as `Llama-3.2-3B-Instruct-Q4_0.gguf` or `Meta-Llama-3-8B-Instruct-Q4_0.gguf`, depending on your needs.
Check models below.

## Download Model Files

Download `Q4_0` and (optionally) `Q8_0` quantized .gguf files from:
- https://huggingface.co/mukel/Llama-3.2-1B-Instruct-GGUF
  - https://huggingface.co/mukel/Llama-3.2-3B-Instruct-GGUF
  - https://huggingface.co/mukel/Meta-Llama-3.1-8B-Instruct-GGUF
  - https://huggingface.co/mukel/Meta-Llama-3-8B-Instruct-GGUF

The `Q4_0` quantized models are recommended, except for the very small models (1B), please be gentle with [huggingface.co](https://huggingface.co) servers:
```
# Llama 3.2 (1B) - Q4_0
curl -L -O https://huggingface.co/mukel/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf

# Llama 3.2 (3B) - Q4_0 
curl -L -O https://huggingface.co/mukel/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf

# Llama 3 (8B) - Q4_0 
curl -L -O https://huggingface.co/mukel/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_0.gguf

# Llama 3.2 (1B) - Q8_0 
curl -L -O https://huggingface.co/mukel/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf

# Llama 3.1 (8B) - Q8_0 
curl -L -O https://huggingface.co/mukel/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_0.gguf

# Llama 3 (8B) - Q8_0
# Optionally download the Q8_0 quantized models
# curl -L -O https://huggingface.co/mukel/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q8_0.gguf
# curl -L -O https://huggingface.co/mukel/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
```

-----------

## Running `llama-tornado`

To execute Llama3 models with TornadoVM on GPUs use the `llama-tornado` script with the `--gpu` flag.

### Usage Examples

#### Basic Inference
Run a model with a text prompt:

```bash
./llama-tornado --gpu --verbose-init --opencl --model Llama-3.2-1B-Instruct-Q8_0.gguf --prompt "Explain the benefits of GPU acceleration."
```

#### GPU Execution (Q8_0 Model)
Enable GPU acceleration with Q8_0 quantization:
```bash
llama-tornado --gpu  --verbose-init --model Llama-3.2-1B-Instruct-Q8_0.gguf --prompt "tell me a joke"
```
#### GPU Execution (Q4_0 Model)
Run with Q4_0 quantization for lower memory usage:
```bash
llama-tornado --gpu  --verbose-init --model Llama-3.2-1B-Instruct-Q4_0.gguf --prompt "tell me a joke"
```

-----------

## Troubleshooting GPU Memory Issues

### Out of Memory Error

You may encounter an out of memory error like:
```
Exception in thread "main" uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException: Unable to allocate 100663320 bytes of memory.
To increase the maximum device memory, use -Dtornado.device.memory=<X>GB
```

This indicates that the default GPU memory allocation (7GB) is insufficient for your model.

### Solution

First, check your GPU specifications. If your GPU has high memory capacity, you can increase the GPU memory allocation using the `--gpu-memory` flag:

```bash
# For 3B models, try increasing to 15GB
./llama-tornado --gpu --model Llama-3.2-3B-Instruct-Q4_0.gguf --prompt "Tell me a joke" --gpu-memory 15GB

# For 8B models, you may need even more (20GB or higher)
./llama-tornado --gpu --model Meta-Llama-3-8B-Instruct-Q4_0.gguf --prompt "Tell me a joke" --gpu-memory 20GB
```

### GPU Memory Requirements by Model Size

| Model Size | Recommended GPU Memory |
|------------|------------------------|
| 1B models  | 7GB (default)          |
| 3B models  | 15GB                   |
| 8B models  | 20GB+                  |

**Note**: If you still encounter memory issues, try:

1. Using Q4_0 instead of Q8_0 quantization (requires less memory).
2. Closing other GPU-intensive applications in your system.

-----------

## Command Line Options

Supported command-line options include:

```bash
cmd ➜ llama-tornado --help
usage: llama-tornado [-h] --model MODEL_PATH [--prompt PROMPT] [-sp SYSTEM_PROMPT] [--temperature TEMPERATURE] [--top-p TOP_P] [--seed SEED] [-n MAX_TOKENS]
                     [--stream STREAM] [--echo ECHO] [-i] [--instruct] [--gpu] [--opencl] [--ptx] [--gpu-memory GPU_MEMORY] [--heap-min HEAP_MIN] [--heap-max HEAP_MAX]
                     [--debug] [--profiler] [--profiler-dump-dir PROFILER_DUMP_DIR] [--print-bytecodes] [--print-threads] [--print-kernel] [--full-dump]
                     [--show-command] [--execute-after-show] [--opencl-flags OPENCL_FLAGS] [--max-wait-events MAX_WAIT_EVENTS] [--verbose]

GPU-accelerated LLaMA.java model runner using TornadoVM

options:
  -h, --help            show this help message and exit
  --model MODEL_PATH    Path to the LLaMA model file (e.g., Llama-3.2-1B-Instruct-Q8_0.gguf) (default: None)

LLaMA Configuration:
  --prompt PROMPT       Input prompt for the model (default: None)
  -sp SYSTEM_PROMPT, --system-prompt SYSTEM_PROMPT
                        System prompt for the model (default: None)
  --temperature TEMPERATURE
                        Sampling temperature (0.0 to 2.0) (default: 0.1)
  --top-p TOP_P         Top-p sampling parameter (default: 0.95)
  --seed SEED           Random seed (default: current timestamp) (default: None)
  -n MAX_TOKENS, --max-tokens MAX_TOKENS
                        Maximum number of tokens to generate (default: 512)
  --stream STREAM       Enable streaming output (default: True)
  --echo ECHO           Echo the input prompt (default: False)

Mode Selection:
  -i, --interactive     Run in interactive/chat mode (default: False)
  --instruct            Run in instruction mode (default) (default: True)

Hardware Configuration:
  --gpu                 Enable GPU acceleration (default: False)
  --opencl              Use OpenCL backend (default) (default: None)
  --ptx                 Use PTX/CUDA backend (default: None)
  --gpu-memory GPU_MEMORY
                        GPU memory allocation (default: 7GB)
  --heap-min HEAP_MIN   Minimum JVM heap size (default: 20g)
  --heap-max HEAP_MAX   Maximum JVM heap size (default: 20g)

Debug and Profiling:
  --debug               Enable debug output (default: False)
  --profiler            Enable TornadoVM profiler (default: False)
  --profiler-dump-dir PROFILER_DUMP_DIR
                        Directory for profiler output (default: /home/mikepapadim/repos/gpu-llama3.java/prof.json)

TornadoVM Execution Verbose:
  --print-bytecodes     Print bytecodes (tornado.print.bytecodes=true) (default: False)
  --print-threads       Print thread information (tornado.threadInfo=true) (default: False)
  --print-kernel        Print kernel information (tornado.printKernel=true) (default: False)
  --full-dump           Enable full debug dump (tornado.fullDebug=true) (default: False)
  --verbose-init        Enable timers for TornadoVM initialization (llama.EnableTimingForTornadoVMInit=true) (default: False)

Command Display Options:
  --show-command        Display the full Java command that will be executed (default: False)
  --execute-after-show  Execute the command after showing it (use with --show-command) (default: False)

Advanced Options:
  --opencl-flags OPENCL_FLAGS
                        OpenCL compiler flags (default: -cl-denorms-are-zero -cl-no-signed-zeros -cl-finite-math-only)
  --max-wait-events MAX_WAIT_EVENTS
                        Maximum wait events for TornadoVM event pool (default: 32000)
  --verbose, -v         Verbose output (default: False)

```

## Debug & Profiling Options
View TornadoVM's internal behavior:
```bash
# Print thread information during execution
./llama-tornado --gpu --model model.gguf --prompt "..." --print-threads

# Show bytecode compilation details
./llama-tornado --gpu --model model.gguf --prompt "..." --print-bytecodes

# Display generated GPU kernel code
./llama-tornado --gpu --model model.gguf --prompt "..." --print-kernel

# Enable full debug output with all details
./llama-tornado --gpu --model model.gguf --prompt "..." --debug --full-dump

# Combine debug options
./llama-tornado --gpu --model model.gguf --prompt "..." --print-threads --print-bytecodes --print-kernel
```

-----------

## Easy Integration with Your Codebase or Tools

To integrate `llama-tornado` into your codebase or IDE (e.g., IntelliJ) or custom build system (like IntelliJ, Maven, or Gradle), use the `--show-command` flag. 
This flag shows the exact Java command with all JVM flags that are being invoked under the hood in order to enable seamless execution on GPUs with TornadoVM.
Hence, it makes it simple to replicate or embed the invoked flags in any external tool or codebase.

```bash
llama-tornado --gpu --model Llama-3.2-1B-Instruct-Q8_0.gguf --prompt "tell me a joke" --show-command
```

<details>
<summary>📋 Click to see the full Java command</summary>

```java
/home/mikepapadim/.sdkman/candidates/java/current/bin/java \
    -server \
    -XX:+UnlockExperimentalVMOptions \
    -XX:+EnableJVMCI \
    -Xms20g -Xmx20g \
    --enable-preview \
    -Djava.library.path=/home/mikepapadim/manchester/TornadoVM/bin/sdk/lib \
    -Djdk.module.showModuleResolution=false \
    --module-path .:/home/mikepapadim/manchester/TornadoVM/bin/sdk/share/java/tornado \
    -Dtornado.load.api.implementation=uk.ac.manchester.tornado.runtime.tasks.TornadoTaskGraph \
    -Dtornado.load.runtime.implementation=uk.ac.manchester.tornado.runtime.TornadoCoreRuntime \
    -Dtornado.load.tornado.implementation=uk.ac.manchester.tornado.runtime.common.Tornado \
    -Dtornado.load.annotation.implementation=uk.ac.manchester.tornado.annotation.ASMClassVisitor \
    -Dtornado.load.annotation.parallel=uk.ac.manchester.tornado.api.annotations.Parallel \
    -Duse.tornadovm=true \
    -Dtornado.threadInfo=false \
    -Dtornado.debug=false \
    -Dtornado.fullDebug=false \
    -Dtornado.printKernel=false \
    -Dtornado.print.bytecodes=false \
    -Dtornado.device.memory=7GB \
    -Dtornado.profiler=false \
    -Dtornado.log.profiler=false \
    -Dtornado.profiler.dump.dir=/home/mikepapadim/repos/gpu-llama3.java/prof.json \
    -Dtornado.enable.fastMathOptimizations=true \
    -Dtornado.enable.mathOptimizations=false \
    -Dtornado.enable.nativeFunctions=fast \
    -Dtornado.loop.interchange=true \
    -Dtornado.eventpool.maxwaitevents=32000 \
    "-Dtornado.opencl.compiler.flags=-cl-denorms-are-zero -cl-no-signed-zeros -cl-finite-math-only" \
    --upgrade-module-path /home/mikepapadim/manchester/TornadoVM/bin/sdk/share/java/graalJars \
    @/home/mikepapadim/manchester/TornadoVM/bin/sdk/etc/exportLists/common-exports \
    @/home/mikepapadim/manchester/TornadoVM/bin/sdk/etc/exportLists/opencl-exports \
    --add-modules ALL-SYSTEM,tornado.runtime,tornado.annotation,tornado.drivers.common,tornado.drivers.opencl \
    -cp /home/mikepapadim/repos/gpu-llama3.java/target/gpu-llama3-1.0-SNAPSHOT.jar \
    com.example.LlamaApp \
    -m Llama-3.2-1B-Instruct-Q8_0.gguf \
    --temperature 0.1 \
    --top-p 0.95 \
    --seed 1746903566 \
    --max-tokens 512 \
    --stream true \
    --echo false \
    -p "tell me a joke" \
    --instruct
```

</details>

-----------

## Current Features & Roadmap

  - **Support for GGUF format models** with Q8_0 and Q4_0 quantization.
  - **Instruction-following and chat modes** for various use cases.
  - **Interactive CLI** with `--interactive` and `--instruct` modes.
  - **Flexible backend switching** - choose OpenCL or PTX at runtime (need to build TornadoVM with both enabled).
  - **Cross-platform compatibility**:
    - ✅ NVIDIA GPUs (OpenCL & PTX )
    - ✅ Intel GPUs (OpenCL)
    - ✅ Apple GPUs (OpenCL)

Click [here](https://github.com/beehive-lab/GPULlama3.java/tree/main/docs/TORNADOVM_TRANSFORMER_OPTIMIZATIONS.md) to view a more detailed list of the transformer optimizations implemented in TornadoVM.

Click [here](https://github.com/beehive-lab/GPULlama3.java/tree/main/docs/GPULlama3_ROADMAP.md) to see the roadmap of the project.

-----------

## License

MIT
