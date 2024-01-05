using simpleNeuralNetwork.Domain;
using simpleNeuralNetwork.Utilities;

await Demo();

static async Task Demo(bool train = false, bool test = true, bool saveNetworkFile = false, bool loadNetworkFile = true)
{
    await FileDownloader.CheckAndDownloadNeededFiles();

    //Initialize untrained network
    var nn = new NeuralNetwork([784, 16, 16, 10], 1, ActivationFunction.ActivationFunctionsType.Sigmoid);

    if (train)
    {
        //Training procedure
        Console.WriteLine("Training...");
        //Load training data
        var trainImages = new MNISTImageFileReader(@".\AppData\train-images.idx3-ubyte");
        var trainLabels = new MNISTLabelFileReader(@".\AppData\train-labels.idx1-ubyte");        

        //Define batch size
        var imagesPerBatch = 5;

        //Do the thing
        for (int i = 0; i < trainImages._numberOfImages / imagesPerBatch; i++)
        {
            if (10 * i % (trainImages._numberOfImages / imagesPerBatch) == 0)
                Console.WriteLine($"{i + 1} / {trainImages._numberOfImages / imagesPerBatch}");

            nn.TrainOnBatch(trainImages.GetNthMiniBatchAsMatrixArray(i, imagesPerBatch), trainLabels.GetNthMiniBatchAsMatrixArray(i, imagesPerBatch));
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
        var testImages = new MNISTImageFileReader(@".\AppData\t10k-images.idx3-ubyte");
        var testLabels = new MNISTLabelFileReader(@".\AppData\t10k-labels.idx1-ubyte");

        //Do the thing
        for (int i = 0; i < 10; i++)
        {
            testImages.PrintNth(i);
            Console.WriteLine($"Expected...\t{testLabels.GetNthLabel(i)}");
            Console.WriteLine($"Extimated...\t{ValoreStimato(nn.FeedForward(testImages.GetNthImageVector(i)))}");
            Console.WriteLine();
        }
    }

}





static int ValoreStimato(Matrix<double> output)
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


