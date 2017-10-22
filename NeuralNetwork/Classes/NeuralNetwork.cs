/// <summary>
/// Notes:
///  Terms:
///     Gamma (Variance between the neural network's output and the expected values)
///     weight (Random weight of the neurons)
///     weightDelta (Error change required in the weight of the neuron)
///     layer (Layers):
///                 input  - basic input into the network
///                 hidden - (neuron) of change to calculate how to get to result
///                 output - output layer
///  
/// </summary>
namespace NeuralNetwork.Classes
{
    public class NeuralNetwork
    {
        /// <summary>
        /// Variables 
        /// </summary>
        private int[] layer;
        private Layer[] layers;
        private int layersCnt = 0;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="layer"></param>
        public NeuralNetwork(int[] layer)
        {
            // Interal Variables
            int len = layer.Length;
            int i = 0;

            // Build and store the layer neuron counts
            this.layer = new int[len];
            for (i = 0; i < len; i++)
                this.layer[i] = layer[i];

            // Setup layer Array 
            // The input layer does NOT need to change. So we can do one less layer
            len--;
            this.layers = new Layer[len];
            this.layersCnt = len;

            // Create a layer of neurons.
            // Using the number of inputs to 'our'(new) layer
            // and the number of outputs to the next layers neuron count
            for (i = 0; i < this.layersCnt; i++)
                this.layers[i] = new Layer(this.layer[i], this.layer[i + 1]);
        }

        /// <summary>
        /// Processer methods for the neural network
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double[] FeedForward(double[] inputs)
        {
            // Feed the input layer with the base inputs
            layers[0].FeedForward(inputs);

            // Iterate over all layers passing the outputs to the next layer
            for (int i = 1; i < this.layersCnt; i++)
                layers[i].FeedForward(layers[i - 1].OutPut);

            // Return previous layers output 
            return layers[this.layersCnt - 1].OutPut;
        }

        public void BackProp(double[] expected)
        {
            // Internal variables
            int Cnt = this.layersCnt - 1;
            int i = 0;

            // Back feed from the output layer
            layers[Cnt].BackPropOutPut(expected);

            // Iterate over the layers going backwards (minus the input layer)
            for (i = Cnt - 1; i >= 1; i--)
            {
                // Going off the forward layer determain what the our gamma and weightsDelta for our 
                // layer based off the gamma and weightsDelta it calculated from the output/upper hidden layer
                layers[i].BackPropHidden(layers[i + 1].Gammas, layers[i + 1].Weights);
            }

            // update the weights (of hidden/output layers)
            for (i = 1; i < this.layersCnt; i++)
                layers[i].UpdateWeights();
        }
    }
}
