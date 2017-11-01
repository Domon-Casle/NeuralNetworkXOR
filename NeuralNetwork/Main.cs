using NeuralNetwork.Learning.XOR;

namespace NeuralNetwork
{
    class NeuralNetworkProgram
    {
        static void Main()
        {
            // XOR Learning
            XORMain xorMain = new XORMain();
            xorMain.Learn(5000);
            xorMain.NNResult();
        }
    }
}
