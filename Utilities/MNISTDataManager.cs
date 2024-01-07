using simpleNeuralNetwork.Extensions;

namespace simpleNeuralNetwork.Utilities
{
    public class MNISTDataManager
    {
        private readonly MNISTImageFileReader _mifr;
        private readonly MNISTLabelFileReader _mlfr;
        public int Cardinality { get; }
        public MNISTDataManager(MNISTImageFileReader mifr, MNISTLabelFileReader mlfr)
        {
            if (mifr._numberOfImages != mlfr._numberOfLabels)
                throw new Exception("The number of images in the provided MNISTImageFileReader must be equal to the number of labels in the provided MNISTLabelFileReader");
            
            _mifr = mifr;
            _mlfr = mlfr;
            Cardinality = mifr._numberOfImages;
        }

        public (Image image, int label) GetNthImageLabelPair(int n)
        {
            return (_mifr.GetNthImage(n), _mlfr.GetNthLabel(n));
        }

        public (Matrix<double> input, Matrix<double> target)[] GetNthMiniBatchAsMatrixPairArray(int n, int samplesPerBatch)
        {
            var result = new (Matrix<double> input, Matrix<double> target)[samplesPerBatch];

            if (n * samplesPerBatch > Cardinality)
                throw new Exception("out of boundaries");

            var (image, label) = (Image.Empty(), 0);

            for (int i = 0; i < samplesPerBatch; i++)
            {
                (image, label) = GetNthImageLabelPair(n * samplesPerBatch + i);
                result[i] = (image.AsVector(), label.ToLabelVector(10));
            }
            return result;
        }
    }
}
