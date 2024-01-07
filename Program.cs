using simpleNeuralNetwork.Domain;
using simpleNeuralNetwork.Utilities;

await Demo();

Console.ReadLine();
static async Task Demo(bool chackMissingFiles = true, bool train = false, bool test = true, bool saveNetworkFile = false, bool loadNetworkFile = true)
{
    if (chackMissingFiles)
        await FileDownloader.CheckAndDownloadNeededFiles();

    //Initialize untrained network
    var nn = new NeuralNetwork([784, 16, 16, 10], 1, ActivationFunction.ActivationFunctionsType.Sigmoid);

    if (train)
    {
        //Training procedure
        Console.WriteLine("Training...");
        //Load training data
        var dataManager = new MNISTDataManager(new MNISTImageFileReader(@".\AppData\train-images.idx3-ubyte"),
                                               new MNISTLabelFileReader(@".\AppData\train-labels.idx1-ubyte"));

        //Define batch size
        var imagesPerBatch = 5;

        //Do the thing
        for (int i = 0; i < dataManager.Cardinality / imagesPerBatch; i++)
        {
            if (10 * i % (dataManager.Cardinality / imagesPerBatch) == 0)
                Console.WriteLine($"{i + 1} / {dataManager.Cardinality / imagesPerBatch}");

            nn.TrainOnBatch(dataManager.GetNthMiniBatchAsMatrixPairArray(i, imagesPerBatch));
        }

        //Potentially save the resulting network in a file for later reuse
        if (saveNetworkFile)
            nn.Save(@".\AppData\nn.dat");
    }

    if (test) 
    {
        //Testing procedure
        Console.WriteLine("Testing...");

        //Potentially load the network from a separate file
        if (loadNetworkFile)
        {
            Console.WriteLine("Loading nn.dat file");
            nn = NeuralNetwork.Load(@".\AppData\nn.dat");
        }

        //Load test data
        var dataManager = new MNISTDataManager(new MNISTImageFileReader(@".\AppData\t10k-images.idx3-ubyte"),
                                               new MNISTLabelFileReader(@".\AppData\t10k-labels.idx1-ubyte"));

        //Do the thing
        for (int i = 0; i < 10; i++)
        {
            var (image, label) = dataManager.GetNthImageLabelPair(i);

            Console.WriteLine(image);
            Console.WriteLine($"Expected...\t{label}");
            Console.WriteLine($"Estimated...\t{EstimatedValue(nn.FeedForward(image.AsVector()))}");
            Console.WriteLine();
        }

    }

}

static (Matrix<double> input, Matrix<double> target)[] GetRandomPairImagesMiniBatch(Random random,
                                                                                    int imagePairsPerBatch,
                                                                                    MNISTImageFileReader mifr,
                                                                                    MNISTLabelFileReader mlfr)
{
    //Extracts Two random images from the mifr, takes their labels from the mlfr, and uses everithing to build the input - target pair
    var result = new (Matrix<double> input, Matrix<double> target)[imagePairsPerBatch];
    var r = 0;
    var firstImage = new Matrix<double>(mifr.PixelNumber, 1);
    var secondImage = new Matrix<double>(mifr.PixelNumber, 1);
    var firstLabel = 0;
    var secondlabel = 0;
    for (int i = 0; i < imagePairsPerBatch; i++)
    {
        r = (int)(random.NextDouble() * mifr._numberOfImages);
        firstImage = mifr.GetNthImageAsDoubleArray(r);
        firstLabel = mlfr.GetNthLabel(r);
        r = (int)(random.NextDouble() * mifr._numberOfImages);
        secondImage = mifr.GetNthImageAsDoubleArray(r);
        secondlabel = mlfr.GetNthLabel(r);

        result[i] = (Append(firstImage, secondImage), firstLabel == secondlabel ? (double[])[1.0, 0.0] : [0.0, 1.0]);
    }

    return result;
}


static Matrix<double> Append(Matrix<double> x, Matrix<double> y)
{
    var result = new Matrix<double>(x.Rows + y.Rows, 1);

    for (int i = 0; i < x.Rows; i++)
    {
        result[i, 0] = x[i, 0];
    }

    for (int i = 0; i < y.Rows; i++)
    {
        result[x.Rows + i, 0] = y[i, 0];
    }

    return result;
}
static int EstimatedValue(Matrix<double> output)
{
    double max = output[0, 0];
    int result = 0;
    for (int i = 0; i < 10; i++)
    {
        if (output[i, 0] > max)
        {
            max = output[i, 0];
            result = i;
        }
    }
    return result;
}



