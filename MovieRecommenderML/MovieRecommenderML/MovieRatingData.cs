using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;


namespace MovieRecommenderML
{
    public class MovieRating
    {
        [LoadColumn(0)]
        public float userId;
        [LoadColumn(1)]
        public float movieId;
        [LoadColumn(2)]
        public float Label;
    }

    public class MovieRatingPrediction
    {
        public float Label;
        public float Score;
    }

}
