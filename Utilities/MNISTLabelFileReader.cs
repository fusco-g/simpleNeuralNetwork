using simpleNeuralNetwork.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace simpleNeuralNetwork.Utilities
{
    //File specification for the mnist dataset are available at http://yann.lecun.com/exdb/mnist/
    public class MNISTLabelFileReader
    {
        private readonly string _labelFilePath;
        private readonly byte[] _data;
        private readonly int _numberOfLabels;
        public MNISTLabelFileReader(string labelFilePath)
        {
            _labelFilePath = labelFilePath;

            using (var labels = new BinaryReader(new FileStream(labelFilePath, FileMode.Open)))
            {

                labels.ReadBigInt32(); //the file starts with an unneeded magic number
                _numberOfLabels = labels.ReadBigInt32();
                _data = labels.ReadBytes(_numberOfLabels);
            }

        }

        public void Print(int n)
        {
            for (int i = 0; i < n && i < _data.Length; i++)
            {
                Console.WriteLine(_data[i].ToString());
            }
        }

        public int GetNthLabel(int n)
        {
            return _data[n];
        }

        public int[] GetNthMiniBatch(int n, int labelsPerBatch)
        {
            var result = new int[labelsPerBatch];

            if (n * labelsPerBatch > _numberOfLabels)
                throw new Exception("out of boundries");

            for (int i = 0; i < labelsPerBatch; i++)
            {
                result[i] = GetNthLabel(n * labelsPerBatch + i);
            }
            return result;
        }
        public Matrix<double>[] GetNthMiniBatchAsMatrixArray(int n, int labelsPerBatch)
        {
            var result = new Matrix<double>[labelsPerBatch];

            if (n * labelsPerBatch > _numberOfLabels)
                throw new Exception("out of boundries");

            for (int i = 0; i < labelsPerBatch; i++)
            {
                result[i] = GetLabelVector(GetNthLabel(n * labelsPerBatch + i));
            }
            return result;
        }

        public double[] GetLabelVector(int n)
        {
            var result = new double[10];

            for (int i = 0; i < 10; i++)
            {
                result[i] = i == n ? 1.0 : 0.0;
            }

            return result;
        }
    }
}
