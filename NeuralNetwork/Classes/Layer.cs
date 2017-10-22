using System;
using NeuralNetwork.Consts;

namespace NeuralNetwork.Classes
{
    public class Layer
    {
        /// <summary>
        /// Variables
        /// </summary>
        private int numberOfInputs; // # of neurons in the previous layer
        private int numberOfOutputs; // # of neurons in the current layer
        public Neuron[] neurons;
        
        /// <summary>
        /// Constructors
        /// </summary>
        /// <param name="numberOfInputs"></param>
        /// <param name="numberOfOutputs"></param>
        public Layer(int numberOfInputs, int numberOfOutputs)
        {
            // Save number of inputs and outputs
            this.numberOfInputs = numberOfInputs;
            this.numberOfOutputs = numberOfOutputs;

            // Build neurons
            neurons = new Neuron[this.numberOfOutputs];
            for (int i = 0; i < this.numberOfOutputs; i++)
                neurons[i] = new Neuron(this.numberOfInputs);
        }

        /// <summary>
        /// Processer methods for the layer
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        // Feed the given inputs to the next layer
        public double[] FeedForward(double[] inputs)
        {
            // Return value
            double[] ffOutput = new double[numberOfOutputs];

            // Run over all our layers neurons 
            for (int i = 0; i < numberOfOutputs; i++)
                ffOutput[i] = neurons[i].FeedForward(inputs);

            return ffOutput;
        }

        /// <summary>
        /// Update the weights of the neurons of this layer
        /// </summary>
        public void UpdateWeights()
        {
            // Update the weights of outputs at inputs for this layer
            for (int i = 0; i < numberOfOutputs; i++)
                neurons[i].UpdateWeight();
        }

        /// <summary>
        /// Due to the calulations of of back proping a previous layer being different from
        /// the calulations for hidden layers this is function handles the output layer.
        /// 
        /// Other notes:
        ///     To train only the output layer can just run this function over and over.
        /// </summary>
        /// <param name="expected"></param>
        public void BackPropOutPut(double[] expected)
        {
            // Caluclate the weightDelta based off the above gamma * our inputs
            for (int i = 0; i < numberOfOutputs; i++)
                neurons[i].BackPropOutput(expected[i]);
        }

        /// <summary>
        /// Due to the calulations of of back proping a previous layer being different from
        /// the calulations for hidden layers this is function handles a hidden layer
        /// </summary>
        public void BackPropHidden(double[] gammaForward, double[,] weightsForward)
        {
            // Caluclate the weightDelta based off the above gamma * our inputs
            for (int i = 0; i < numberOfOutputs; i++)
                neurons[i].BackPropHidden(gammaForward, weightsForward, i);
        }

        /// <summary>
        /// Get the neuron outputs
        /// </summary>
        /// <returns></returns>
        public double[] OutPut
        {
            get
            {
                double[] temp = new double[numberOfOutputs];
                for (int i = 0; i < numberOfOutputs; i++)
                    temp[i] = neurons[i].output;

                return temp;
            }
            private set { }
        }

        /// <summary>
        /// Get the neuron Gamma values
        /// </summary>
        /// <returns></returns>
        public double[] Gammas
        {
            get
            {
                double[] temp = new double[numberOfOutputs];
                for (int i = 0; i < numberOfOutputs; i++)
                    temp[i] = neurons[i].gamma;

                return temp;
            }
            private set { }
        }

        /// <summary>
        /// Get the neuron weights values
        /// </summary>
        /// <returns></returns>
        public double[,] Weights
        {
            get
            {
                double[,] temp = new double[numberOfOutputs, numberOfInputs];
                for (int i = 0; i < numberOfOutputs; i++)
                {
                    for (int j = 0; j < numberOfInputs; j++)
                        temp[i, j] = neurons[i].weights[j];
                }

                return temp;
            }
            private set { }
        }
    }
}
