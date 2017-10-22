using System;
using System.Security.Cryptography;

namespace NeuralNetwork.Consts
{
    public class CryptoRandom
    {
        /// <summary>
        /// Constructor for cryptoRandom
        /// </summary>
        public CryptoRandom()
        {
        }

        /// <summary>
        /// Get/Set of the random value used by neurons
        /// </summary>
        public double RandomValue
        {
            get
            {
                using (RNGCryptoServiceProvider p = new RNGCryptoServiceProvider())
                {
                    Random r = new Random(p.GetHashCode());
                    return r.NextDouble();
                }
            }
            private set { }
        }
    }
}
