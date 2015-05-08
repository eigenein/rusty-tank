//! SVD implementation.
//!
//! See http://habrahabr.ru/company/surfingbird/blog/141959/.

#[derive(Debug)]
pub struct Model {
    /// Base predictor.
    base: f32,
    /// Base row predictors.
    row_bases: Vec<f32>,
    /// Base column predictors.
    column_bases: Vec<f32>,
    /// Row features.
    row_features: Vec<Vec<f32>>,
    /// Column features.
    column_features: Vec<Vec<f32>>,
}

impl Model {
    /// Creates a new model.
    pub fn new(row_count: usize, column_count: usize, feature_count: usize) -> Model {
        Model {
            base: 0.0,
            row_bases: vec![0.0; row_count],
            column_bases: vec![0.0; column_count],
            row_features: Model::new_feature_vectors(row_count, feature_count),
            column_features: Model::new_feature_vectors(column_count, feature_count),
        }
    }

    /// Makes a step.
    pub fn make_step() {
        // TODO.
    }

    /// Creates a vector of feature vectors.
    fn new_feature_vectors(count: usize, feature_count: usize) -> Vec<Vec<f32>> {
        (0..count).map(|_| vec![0.0; feature_count]).collect()
    }
}
