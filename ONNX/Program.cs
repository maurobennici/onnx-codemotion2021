using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ONNX
{
    class Program
    {
        static void Main(string[] args)
        {
            // data pre-processing
            // REMEMBER the tokenizer + vocab
            // see the project https://github.com/Microsoft/BlingFire
            var t = new long[] {
                102, 4714, 395, 1538, 2692, 103, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0 };

            // create the tensor
            var input = new DenseTensor<long>(new[] { 1, 128 });
            for (int i = 0; i < 128; i++)
            {
                input[0, i] = t[i];
            }
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", input)
            };

            // sentiments classification
            var modelFilePath = "c:\\temp\\BERTsentiment.onnx";
            using var session = new InferenceSession(modelFilePath);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // show result
            var values = (DenseTensor<float>)results.First().Value;
            for (var i = 0; i < values.Length; i++)
            {
                Console.WriteLine(values[0,i]);
            }
            Console.ReadKey();
        }
    }
}
