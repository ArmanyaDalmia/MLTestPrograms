using Microsoft.ML.Data;

namespace ObjectDetectionML.DataStructures
{
    public class ImageNetPrediction
    {
        [ColumnName("grid")]
        public float[] PredictedLabels;
    }
}
