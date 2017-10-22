using System;

namespace NeuralNetwork
{
    class NeuralNetworkProgram
    {
        static void Main()
        {
            // Build the neuralnetwork input
            // int[0] - input
            // int[1 -> (n - 2)] - hidden layers
            // int[n] - output layer
            int[] layers = new int[] {3, 25, 25, 1};

            // XOR learning network...
            // 0 0 0 => 0
            // 0 0 1 => 1
            // 0 1 0 => 1
            // 0 1 1 => 0
            // 1 0 0 => 1
            // 1 0 1 => 0
            // 1 1 0 => 0
            // 1 1 1 => 1 

            Classes.NeuralNetwork thisNeuralNetwork = new Classes.NeuralNetwork(layers);

            // Let it Learn!!! 
            // Learning process
            // FeedForward the what the network thinks is the result
            // Then feed back the error/gamma change
            for (int i = 0; i < 5000; i++)
            {
                // 0 0 0 => 0
                thisNeuralNetwork.FeedForward(new double[] { 0, 0, 0 });
                thisNeuralNetwork.BackProp(new double[] { 0 });

                // 0 0 1 => 1
                thisNeuralNetwork.FeedForward(new double[] { 0, 0, 1 });
                thisNeuralNetwork.BackProp(new double[] { 1 });

                // 0 1 0 => 1
                thisNeuralNetwork.FeedForward(new double[] { 0, 1, 0 });
                thisNeuralNetwork.BackProp(new double[] { 1 });

                // 0 1 1 => 0
                thisNeuralNetwork.FeedForward(new double[] { 0, 1, 1 });
                thisNeuralNetwork.BackProp(new double[] { 0 });

                // 1 0 0 => 1
                thisNeuralNetwork.FeedForward(new double[] { 1, 0, 0 });
                thisNeuralNetwork.BackProp(new double[] { 1 });

                // 1 0 1 => 0
                thisNeuralNetwork.FeedForward(new double[] { 1, 0, 1 });
                thisNeuralNetwork.BackProp(new double[] { 0 });

                // 1 1 0 => 0
                thisNeuralNetwork.FeedForward(new double[] { 1, 1, 0 });
                thisNeuralNetwork.BackProp(new double[] { 0 });

                // 1 1 1 => 1 
                thisNeuralNetwork.FeedForward(new double[] { 1, 1, 1 });
                thisNeuralNetwork.BackProp(new double[] { 1 });
            }

            // Did it learn?
            Console.WriteLine(BuildOutputString(thisNeuralNetwork.FeedForward(new double[] { 0, 0, 0 })[0], 0));
            Console.WriteLine(BuildOutputString(thisNeuralNetwork.FeedForward(new double[] { 0, 0, 1 })[0], 1));
            Console.WriteLine(BuildOutputString(thisNeuralNetwork.FeedForward(new double[] { 0, 1, 0 })[0], 1));
            Console.WriteLine(BuildOutputString(thisNeuralNetwork.FeedForward(new double[] { 0, 1, 1 })[0], 0));
            Console.WriteLine(BuildOutputString(thisNeuralNetwork.FeedForward(new double[] { 1, 0, 0 })[0], 1));
            Console.WriteLine(BuildOutputString(thisNeuralNetwork.FeedForward(new double[] { 1, 0, 1 })[0], 0));
            Console.WriteLine(BuildOutputString(thisNeuralNetwork.FeedForward(new double[] { 1, 1, 0 })[0], 0));
            Console.WriteLine(BuildOutputString(thisNeuralNetwork.FeedForward(new double[] { 1, 1, 1 })[0], 1));

            Console.ReadKey();
        }

        public static string BuildOutputString (double nnOutput, int RealValue)
        {
            return string.Format("Neural Network result: {0}\nExpected Result:       {1}\n", Math.Round(nnOutput).ToString(), RealValue.ToString());
        }
    }
}
