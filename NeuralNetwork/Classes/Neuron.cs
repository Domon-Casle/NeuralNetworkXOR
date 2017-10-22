using System;
using NeuralNetwork.Consts;

namespace NeuralNetwork.Classes
{
    public class Neuron
    {
        /// <summary>
        /// Variables
        /// </summary>
        private int numberOfInputs;
        private double[] inputs;
        private double error;
        private double[] weightDelta;
        public double output;
        public double gamma;
        public double[] weights;

        /// <summary>
        /// Constructors
        /// </summary>
        /// <param name="numberOfInputs"></param>
        /// <param name="numberOfOutputs"></param>
        public Neuron (int numberOfInputs)
        {
            // Build the link to inputs
            this.numberOfInputs = numberOfInputs;
            inputs = new double[this.numberOfInputs];
            weights = new double[this.numberOfInputs];
            weightDelta = new double[this.numberOfInputs];

            // Init the Weight
            InitilizeWeight();
        }

        /// <summary>
        /// Initilize the weight
        /// </summary>
        private void InitilizeWeight()
        {
            // Init the weight to a random number - between (-0.5 to 0.5)
            for (int i = 0; i < this.numberOfInputs; i++)
                weights[i] = (NeuralNetworkConsts.random.RandomValue - 0.5f);
        }

        /// <summary>
        /// Tan deritative of given value
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        private double TanHDer(double value)
        {
            return 1 - (value * value);
        }

        /// <summary>
        /// Figure out our neuron output
        /// </summary>
        /// <param name="input"></param>
        public double FeedForward(double[] inputs)
        {
            // Set our input to incoming inputs
            this.inputs = inputs;

            // Init the output to 0
            output = 0;

            // Run over the input from the previous layer
            // Our output is the inputs times our random weights
            for (int i = 0; i < numberOfInputs; i++)
                output += (inputs[i] * weights[i]);
            
            // Now TanH the output
            output = Math.Tanh(output);

            // Return our final output
            return output;
        }

        /// <summary>
        /// Update our weight based on the Delta error and learning rate
        /// </summary>
        public void UpdateWeight()
        {
            // New weight is weight -= (DeltaWeight * learning rate)
            // AKA Delta weight of previous changes that occured * how fast we think
            // the neuron should learn.
            for (int i = 0; i < numberOfInputs; i++)
                weights[i] -= (weightDelta[i] * NeuralNetworkConsts.LearningRate);
        }

        /// <summary>
        /// Due to the calulations of back proping a previous layer being different from
        /// the calulations for hidden layers this is function handles the output layer.
        /// 
        /// Other notes:
        ///     To train only the output layer just run this function over and over.
        /// </summary>
        /// <param name="expected"></param>
        public void BackPropOutput(double expected)
        {
            // Error is the output we gave vs the expected
            error = output - expected;

            // Gamma is the error times the TanH Derivative of our output
            gamma = error * TanHDer(output);

            // Calculate the weightDelta (gamma * the intput we got)
            for (int i = 0; i < numberOfInputs; i++)
                weightDelta[i] = gamma * inputs[i];
        }

        /// <summary>
        /// Due to the calulations of of back proping a previous layer being different from
        /// the calulations for hidden layers this is function handles a hidden layer
        /// </summary>
        /// <param name="gammaForward"></param>
        /// <param name="weightsForward"></param>
        public void BackPropHidden(double[] gammaForward, double[,] weightsForward, int index)
        {
            // Our gamma is equal to the gamma values of our forward layer * its weights
            int len = gammaForward.Length;
            int i = 0;

            // Default gamma for this back prop to 0
            gamma = 0;
            for (i = 0; i < len; i++)
                gamma += gammaForward[i] * weightsForward[i, index];

            // Finnaly take the gamma * the Tan Deriviative of our output
            gamma *= TanHDer(output);

            // Calculate the weightDelta (gamma * inputs)
            for (i = 0; i < numberOfInputs; i++)
                weightDelta[i] = gamma * inputs[i];
        }
    }
}
