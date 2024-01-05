using simpleNeuralNetwork.Extensions;
using simpleNeuralNetwork.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace simpleNeuralNetwork.Domain
{
    public class NeuralLayer
    {
        public Matrix<double> _weights;
        public Matrix<double> _biases;
        public NeuralLayer(int size, int previousLayesSize)
        {
            var random = new Random(42);
            _weights = random.NextDoubleMatrix(size, previousLayesSize);
            _biases = Matrix<double>.Zeros(size, 1);
        }
        public Matrix<double> FeedForward(Matrix<double> input, ActivationFunction func)
        {
            return (_weights * input + _biases).Map(func._func);
        }
    }
}
