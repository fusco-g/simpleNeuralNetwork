using simpleNeuralNetwork.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace simpleNeuralNetwork.Extensions
{
    public static class IntExtensions
    {
        public static Matrix<double> ToLabelVector(this int n, int vectorSize)
        {
            if (n < 0) throw new ArgumentOutOfRangeException(nameof(n));
            if (vectorSize < n) throw new ArgumentOutOfRangeException(nameof(vectorSize));

            var result = new double[vectorSize];

            for (int i = 0; i < vectorSize; i++)
            {
                result[i] = i == n ? 1.0 : 0.0;
            }

            return result;
        }
    }
}
