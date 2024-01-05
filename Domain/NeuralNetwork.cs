using simpleNeuralNetwork.Extensions;
using simpleNeuralNetwork.Utilities;

namespace simpleNeuralNetwork.Domain
{
    public class NeuralNetwork
    {
        public readonly int[] _layersSizes;
        public readonly double _learningRate;
        private readonly ActivationFunction _activationFunc;
        private readonly ActivationFunction.ActivationFunctionsType _activationFuncType;
        public NeuralLayer[] _layers;
        public NeuralNetwork(int[] layersSizes, double learningRate, ActivationFunction.ActivationFunctionsType activationFuncType)
        {
            _layersSizes = layersSizes;
            _layers = new NeuralLayer[layersSizes.Length - 1];
            _learningRate = learningRate;
            _activationFuncType = activationFuncType;
            _activationFunc = ActivationFunction.Common[activationFuncType];

            for (int i = 1; i < _layersSizes.Length; i++)
            {
                _layers[i - 1] = new NeuralLayer(_layersSizes[i], _layersSizes[i - 1]);
            }
        }

        public Matrix<double> FeedForward(Matrix<double> input)
        {
            var inputCopy = input.Copy();

            foreach (var layer in _layers)
            {
                inputCopy = layer.FeedForward(inputCopy, _activationFunc);
            }

            return inputCopy;
        }

        public void TrainOnBatch(Matrix<double>[] inputs, Matrix<double>[] targets)
        {
            if (inputs.Length != targets.Length)
                throw new ArgumentException("inputs batch size must be equal to targets array size");

            var deltaW = new Matrix<double>[_layers.Length];
            var deltaB = new Matrix<double>[_layers.Length];

            for (int i = 0; i < inputs.Length; i++)
            {
                var (tempDeltaW, tempDeltaB) = Backpropagate(inputs[i], targets[i]);
                for (int j = 0; j < _layers.Length; j++)
                {
                    deltaW[j] = (deltaW[j] ?? Matrix<double>.Zeros(tempDeltaW[j].Rows, tempDeltaW[j].Cols)) + tempDeltaW[j];
                    deltaB[j] = (deltaB[j] ?? Matrix<double>.Zeros(tempDeltaB[j].Rows, tempDeltaB[j].Cols)) + tempDeltaB[j];
                }
            }

            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i]._weights -= _learningRate * deltaW[i] * (1.0 / inputs.Length);
                _layers[i]._biases -= _learningRate * deltaB[i] * (1.0 / inputs.Length);
            }
        }

        public (Matrix<double>[] deltaW, Matrix<double>[] deltaB) Backpropagate(Matrix<double> input, Matrix<double> target)
        {
            var deltaW = new Matrix<double>[_layers.Length];
            var deltaB = new Matrix<double>[_layers.Length];

            var zs = new Matrix<double>[_layers.Length];
            var activations = new Matrix<double>[_layers.Length + 1];
            activations[0] = input;
            for (int i = 0; i < _layers.Length; i++)
            {
                zs[i] = _layers[i]._weights * activations[i] + _layers[i]._biases;
                activations[i + 1] = zs[i].Map(_activationFunc._func);
            }

            var nabla = zs[^1].Map(_activationFunc._dFunc).ElementwiseMult(activations[^1] - target);
            deltaW[^1] = nabla * !activations[^2];
            deltaB[^1] = nabla;

            for (int i = 2; i <= _layers.Length; i++)
            {
                nabla = zs[^i].Map(_activationFunc._dFunc).ElementwiseMult(!_layers[^(i - 1)]._weights * nabla);
                deltaW[^i] = nabla * !activations[^(i + 1)];
                deltaB[^i] = nabla;
            }

            return (deltaW, deltaB);
        }

        public void Save(string filePath)
        {
            using (var nn = new BinaryWriter(new FileStream(filePath, FileMode.OpenOrCreate)))
            {
                nn.Write(_layersSizes.Length);
                for (int i = 0; i < _layersSizes.Length; i++)
                {
                    nn.Write(_layersSizes[i]);
                }
                nn.Write(_learningRate);
                nn.Write((int)_activationFuncType);
                for (int k = 0; k < _layers.Length; k++)
                {
                    nn.Write(_layers[k]._weights.Rows);
                    nn.Write(_layers[k]._weights.Cols);

                    for (int i = 0; i < _layers[k]._weights.Rows; i++)
                    {
                        for (int j = 0; j < _layers[k]._weights.Cols; j++)
                        {
                            nn.Write(_layers[k]._weights[i, j]);
                        }
                    }

                    nn.Write(_layers[k]._biases.Rows);

                    for (int i = 0; i < _layers[k]._biases.Rows; i++)
                    {
                        nn.Write(_layers[k]._biases[i, 0]);
                    }
                }
            }
        }
        public static NeuralNetwork Load(string filePath)
        {
            using (var nn = new BinaryReader(new FileStream(filePath, FileMode.Open)))
            {
                int layersSizesLength = nn.ReadLittleInt32();
                var layersSizes = new int[layersSizesLength];

                for (int i = 0; i < layersSizesLength; i++)
                {
                    layersSizes[i] = nn.ReadLittleInt32();
                }

                var learningRate = nn.ReadLittleDouble();
                var activationFuncType = (ActivationFunction.ActivationFunctionsType)nn.ReadLittleInt32();
                var result = new NeuralNetwork(layersSizes,
                                               learningRate,
                                               activationFuncType);

                for (int k = 0; k < result._layers.Length; k++)
                {
                    var Rows = nn.ReadLittleInt32();
                    var Cols = nn.ReadLittleInt32();
                    result._layers[k]._weights = new Matrix<double>(Rows, Cols);

                    for (int i = 0; i < result._layers[k]._weights.Rows; i++)
                    {
                        for (int j = 0; j < result._layers[k]._weights.Cols; j++)
                        {
                            result._layers[k]._weights[i, j] = nn.ReadLittleDouble();
                        }
                    }
                    Rows = nn.ReadLittleInt32();
                    result._layers[k]._biases = new Matrix<double>(Rows, 1);

                    for (int i = 0; i < result._layers[k]._biases.Rows; i++)
                    {
                        result._layers[k]._biases[i, 0] = nn.ReadLittleDouble();
                    }
                }

                return result;
            }
        }
    }
}
