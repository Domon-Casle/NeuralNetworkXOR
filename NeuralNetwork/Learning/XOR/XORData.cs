using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Learning.XOR
{
    class XORData
    {
        /// <summary>
        /// Variables
        /// </summary>
        private double[] inputs;
        private double[] output;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="output"></param>
        public XORData(double[] inputs, double[] output)
        {
            this.inputs = inputs;
            this.output = output;
        }

        /// <summary>
        /// Access methods
        /// </summary>
        public double[]Input {
            get { return inputs; }
            private set { }
        }

        public double[] Output
        {
            get { return output; }
            private set { }
        }
    }
}
