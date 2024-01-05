using simpleNeuralNetwork.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace simpleNeuralNetwork.Extensions
{
    public static class RandomExtensions
    {
        public static Matrix<double> NextDoubleMatrix(this Random r, int rows, int cols)
        {
            var result = new Matrix<double>(rows, cols);
            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < result.Cols; j++)
                {
                    result[i, j] = r.NextDouble() * 2 - 1;
                }
            }
            return result;
        }
    }
}
