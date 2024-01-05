using simpleNeuralNetwork.Extensions;
using System.IO.Compression;
using System.Text;

namespace simpleNeuralNetwork.Utilities
{
    public class FileDownloader
    {
        public static async Task DownloadAsync(string fileUrl,
                                               string filePath,
                                               string? decompressedFilePath = null,
                                               CancellationToken cancellationToken = default)
        {
            using (var client = new HttpClient())
            {
                try
                {
                    Console.WriteLine("Downloading...");

                    using var response = await client.GetAsync(fileUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                    var contentLength = response.Content.Headers.ContentLength;
                    using var download = await response.Content.ReadAsStreamAsync(cancellationToken);
                    using var file = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
                    var buffer = new byte[8192];
                    long totalBytesRead = 0;
                    int bytesRead;

                    while ((bytesRead = await download.ReadAsync(buffer, 0, buffer.Length, cancellationToken).ConfigureAwait(false)) != 0)
                    {
                        await file.WriteAsync(buffer, 0, bytesRead, cancellationToken).ConfigureAwait(false);
                        totalBytesRead += bytesRead;
                    }

                    Console.WriteLine("\nDownload of the .gz file compleated, now the file will be decompressed");
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine("Download cancelled.");
                }
                catch (Exception e)
                {
                    Console.WriteLine($"{e.Message}");
                    throw;
                }

            }

            DecompressFile(filePath, decompressedFilePath);
            File.Delete(filePath);
        }

        private static void DecompressFile(string CompressedFilePath, string? DecompressedFilePath = null)
        {
            if (CompressedFilePath == null)
                throw new ArgumentNullException(nameof(CompressedFilePath));

            DecompressedFilePath = !string.IsNullOrWhiteSpace(DecompressedFilePath)
                                 ? DecompressedFilePath
                                 : Path.Combine(@".\AppData\", GetGZCompressedFileName(CompressedFilePath) ?? "newfile");

            using FileStream compressedFileStream = File.Open(CompressedFilePath, FileMode.Open);
            using FileStream outputFileStream = File.Create(DecompressedFilePath);

            compressedFileStream.Position = 0;
            using var decompressor = new GZipStream(compressedFileStream, CompressionMode.Decompress);
            decompressor.CopyTo(outputFileStream);
        }

        private static string? GetGZCompressedFileName(string CompressedFilePath)
        {
            var result = new StringBuilder();
            using var binaryReader = new BinaryReader(new FileStream(CompressedFilePath, FileMode.Open));
            binaryReader.ReadBytes(3); //skip ID1 (IDentification 1), ID2(IDentification 2), CM (Compression Method)
            byte flag = binaryReader.ReadByte();
            binaryReader.ReadBytes(6); //skip MTIME (Modification TIME), XFL (eXtra FLags), OS (Operating System)

            if ((flag & 0b100) > 0) //if FEXTRA flag is set
            {
                short xLen = binaryReader.ReadLittleInt16();
                binaryReader.ReadBytes(xLen); //skip xLen bytes of "extra field"
            }

            if ((flag & 0b1000) > 0)
            {
                char c;
                while ((c = (char)binaryReader.ReadByte()) != '\0')
                {
                    result.Append(c);
                }
                return result.ToString();
            }
            return null;
        }

        public static async Task CheckAndDownloadNeededFiles()
        {
            var localPath = @".\AppData";
            var neededFiles = new List<(string url, string localFileName)>
            {
                (@"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", @"t10k-images.idx3-ubyte"),
                (@"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", @"t10k-labels.idx1-ubyte"),
                (@"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", @"train-images.idx3-ubyte"),
                (@"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", @"train-labels.idx1-ubyte"),
            };

            foreach (var neededFile in neededFiles)
            {
                if (!File.Exists(Path.Combine(localPath, neededFile.localFileName)))
                {
                    Console.WriteLine($"The {neededFile.localFileName} File is missing, would you like to download it? (y/n)");

                    if ((await Console.In.ReadLineAsync()) == "y")
                        await DownloadAsync(neededFile.url,
                                                      Path.Combine(localPath, neededFile.localFileName.Replace('.', '-') + ".gz"),
                                                      Path.Combine(localPath, neededFile.localFileName));
                }
            }
        }
    }
}
