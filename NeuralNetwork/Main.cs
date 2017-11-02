using NeuralNetwork.Learning.XOR;
using System;

namespace NeuralNetwork
{
    class NeuralNetworkProgram
    {
        static void Main()
        {
            // XOR Learning
            DateTime startDate = DateTime.Now;
            XORMain xorMain = new XORMain();
            xorMain.Learn(5000);
            DateTime endDate = DateTime.Now;
            xorMain.NNResult(startDate, endDate);
        }
    }
}
