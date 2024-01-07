using simpleNeuralNetwork.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace simpleNeuralNetwork.Utilities
{
    //File specification for the mnist dataset are available at http://yann.lecun.com/exdb/mnist/
    public class MNISTImageFileReader
    {
        private readonly string _imageFilePath;
        public readonly int _width;
        public readonly int _height;
        private readonly double[] _data;
        public readonly int _numberOfImages;

        public int PixelNumber => _width * _height;
        public MNISTImageFileReader(string imageFilePath)
        {
            _imageFilePath = imageFilePath;

            using (var images = new BinaryReader(new FileStream(_imageFilePath, FileMode.Open)))
            {
                images.ReadBigInt32(); //the file starts with an unneeded magic number
                _numberOfImages = images.ReadBigInt32();
                _width = images.ReadBigInt32();
                _height = images.ReadBigInt32();
                _data = images.ReadBytes(PixelNumber * _numberOfImages)
                              .Select(x => (double)x / byte.MaxValue)
                              .ToArray();
            }

        }

        public double[] GetNthImageAsDoubleArray(int n)
        {
            return _data.Skip(n * PixelNumber).Take(PixelNumber).ToArray();
        }

        public Image GetNthImage(int n) => new(GetNthImageAsDoubleArray(n), _width, _height);

        public double[][] GetNthMiniBatch(int n, int imagesPerBatch)
        {
            var result = new double[imagesPerBatch][];

            if (n * imagesPerBatch > _numberOfImages)
                throw new Exception("out of boundaries");

            for (int i = 0; i < imagesPerBatch; i++)
            {
                result[i] = GetNthImageAsDoubleArray(n * imagesPerBatch + i);
            }
            return result;
        }

        public Matrix<double>[] GetNthMiniBatchAsMatrixArray(int n, int imagesPerBatch)
        {
            var result = new Matrix<double>[imagesPerBatch];

            if (n * imagesPerBatch > _numberOfImages)
                throw new Exception("out of boundaries");

            for (int i = 0; i < imagesPerBatch; i++)
            {
                result[i] = GetNthImageAsDoubleArray(n * imagesPerBatch + i);
            }
            return result;
        }
    }
}
