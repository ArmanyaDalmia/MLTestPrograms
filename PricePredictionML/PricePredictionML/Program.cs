using System;
using Microsoft.ML;
using PricePredictionMLML.Model.DataModels;

namespace PricePredictionML
{
    class Program
    {
        static void Main(string[] args)
        {
            ConsumeModel();
        }


        public static void ConsumeModel()
        {
            // Load the model
            MLContext mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load("MLModel.zip", out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            // Use the code below to add input data
            var input = new ModelInput()
            {
                Vendor_id = "VTS",
                Rate_code = 1,
                Passenger_count = 1,
                Trip_time_in_secs = 1140,
                Trip_distance = 3.75f,
                Payment_type = "CRD",
                Fare_amount = 0
            };

            // Try model on sample data
            ModelOutput result = predEngine.Predict(input);
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {result.Score:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }

}
}
