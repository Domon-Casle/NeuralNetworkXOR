using NeuralNetwork.Learning.XOR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Learning.XOR
{
    class XORMain
    {
        /// <summary>
        /// Variables
        /// </summary>
        private int[] layers;
        private Classes.NeuralNetwork myNN;
        private List<XORData> data;

        /// <summary>
        /// Constructor
        /// </summary>
        public XORMain ()
        {
            layers = new int[] { 3, 25, 25, 1 };
            myNN = new Classes.NeuralNetwork(layers);
            this.BuildData();
        }

        /// <summary>
        /// Build the 'data'
        /// </summary>
        private void BuildData()
        {
            data = new List<XORData>();
            data.Add(new XORData(new double[] { 0, 0, 0 }, new double[] { 0 }));
            data.Add(new XORData(new double[] { 0, 0, 1 }, new double[] { 1 }));
            data.Add(new XORData(new double[] { 0, 1, 0 }, new double[] { 1 }));
            data.Add(new XORData(new double[] { 0, 1, 1 }, new double[] { 0 }));
            data.Add(new XORData(new double[] { 1, 0, 0 }, new double[] { 1 }));
            data.Add(new XORData(new double[] { 1, 0, 1 }, new double[] { 0 }));
            data.Add(new XORData(new double[] { 1, 1, 0 }, new double[] { 0 }));
            data.Add(new XORData(new double[] { 1, 1, 1 }, new double[] { 1 }));
        }

        public void Learn(int NumberOfTimes)
        {
            for (int i = 0; i < NumberOfTimes; i++)
            {
                for (int j = 0, len = data.Count; j < len; j++)
                {
                    this.myNN.FeedForward(data[j].Input);
                    this.myNN.BackProp(data[j].Output);
                }
            }
        }

        public void NNResult()
        {
            double[] nnOutput;
            string outputString;

            for (int i = 0, len = data.Count; i < len; i++)
            {
                nnOutput = this.myNN.FeedForward(data[i].Input);
                outputString = BuildOutputString(nnOutput[0], data[i].Output[0]);
                Console.WriteLine(outputString);
            }

            Console.Write("Press any key to continue...");
            Console.ReadKey();
        }

        public string BuildOutputString(double nnOutput, double RealValue)
        {
            return string.Format("Neural Network result: {0}\nExpected Result:       {1}\n", Math.Round(nnOutput).ToString(), RealValue.ToString());
        }
    }
}
