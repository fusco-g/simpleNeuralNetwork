using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace simpleNeuralNetwork.Utilities
{
    public class Image(double[] data, int width, int height)
    {
        private readonly double[] _data = data;
        private readonly int _width = width;
        private readonly int _height = height;

        public override string ToString() => ToString();
        public string ToString(char[]? grayscale = null)
        {
            grayscale ??= [' ', '░', '▒', '▓', '█'];

            var stringBuilder = new StringBuilder();

            for (int i = 0; i < _data.Length; i++)
            {
                if (i != 0 && i % _width == 0)
                    stringBuilder.AppendLine();

                stringBuilder.Append(grayscale[(int)Math.Min(_data[i] * grayscale.Length, grayscale.Length - 1)]);
            }

            return stringBuilder.ToString();
        }
        public Matrix<double> AsVector()
        {
            return _data;
        }
        public static Image Empty()
        {
            return new Image(Array.Empty<double>(), 0, 0);
        }
    }
}
