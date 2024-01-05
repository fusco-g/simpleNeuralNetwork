namespace simpleNeuralNetwork.Domain
{
    public class ActivationFunction(Func<double, double> func, Func<double, double> dFunc)
    {
        public Func<double, double> _func = func;
        public Func<double, double> _dFunc = dFunc;
        public enum ActivationFunctionsType
        {
            Sigmoid,
            ReLu
        }

        public static readonly Dictionary<ActivationFunctionsType, ActivationFunction> Common = new()
        {
            {
                ActivationFunctionsType.Sigmoid, 
                new ActivationFunction(x => 1 / (1 + Math.Exp(-x)),
                                       x => Math.Exp(x)/Math.Pow(Math.Exp(x) + 1, 2))
            },
            {
                ActivationFunctionsType.ReLu,
                new ActivationFunction(x => Math.Max(0, x),
                                       x => x switch { (<= 0) => 0, _ => 1 }) 
            }
        };
    }
}
