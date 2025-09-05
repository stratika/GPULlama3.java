package org.beehive.gpullama3;

import org.beehive.gpullama3.aot.AOT;
import org.beehive.gpullama3.auxiliary.LastRunMetrics;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.inference.sampler.CategoricalSampler;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.sampler.ToppSampler;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.loader.ModelLoader;
import org.beehive.gpullama3.tornadovm.FloatArrayUtils;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.IOException;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.beehive.gpullama3.inference.sampler.Sampler.createSampler;
public class LlamaApp {
    // Configuration flags for hardware acceleration and optimizations
    public static final boolean USE_VECTOR_API = Boolean.parseBoolean(System.getProperty("llama.VectorAPI", "true"));   // Enable Java Vector API for CPU acceleration
    public static final boolean USE_AOT = Boolean.parseBoolean(System.getProperty("llama.AOT", "false"));               // Use Ahead-of-Time compilation
    public static final boolean SHOW_PERF_INTERACTIVE = Boolean.parseBoolean(System.getProperty("llama.ShowPerfInteractive", "true")); // Show performance metrics in interactive mode


    /**
     * Loads the language model based on the given options.
     * <p>
     * If Ahead-of-Time (AOT) mode is enabled, attempts to use a pre-loaded compiled model. Otherwise, loads the model from the specified path using the model loader.
     * </p>
     *
     * @param options
     *         the parsed CLI options containing model path and max token limit
     * @return the loaded {@link Model} instance
     * @throws IOException
     *         if the model fails to load
     * @throws IllegalStateException
     *         if AOT loading is enabled but the preloaded model is unavailable
     */
    private static Model loadModel(Options options) throws IOException {
        if (USE_AOT) {
            Model model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
            if (model == null) {
                throw new IllegalStateException("Failed to load precompiled AOT model.");
            }
            return model;
        }
        return ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true, options.useTornadovm());
    }

    private static Sampler createSampler(Model model, Options options) {
        return selectSampler(model.configuration().vocabularySize(), options.temperature(), options.topp(), options.seed());
    }

    private static void runSingleInstruction(Model model, Sampler sampler, Options options) {
        String response = model.runInstructOnce(sampler, options);
        System.out.println(response);
        if (SHOW_PERF_INTERACTIVE) {
            LastRunMetrics.printMetrics();
        }
    }

    /**
     * Entry point for running the LLaMA-based model with provided command-line arguments.
     *
     * <p>Initializes model options, loads the appropriate model (either AOT or on-demand),
     * configures the sampler, and runs either in interactive or single-instruction mode based on the input options.</p>
     *
     * @param args
     *         command-line arguments used to configure model path, temperature, seed, etc.
     * @throws IOException
     *         if model loading or file operations fail.
     */
    static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Model model = loadModel(options);
        Sampler sampler = createSampler(model, options);

        if (options.interactive()) {
            model.runInteractive(sampler, options);
        } else {
            runSingleInstruction(model, sampler, options);
        }
    }
}



