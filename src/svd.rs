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
    row_features: Vec<f32>,
    /// Column features.
    column_features: Vec<f32>
}

impl Model {
    pub fn new(rows: usize, columns: usize) -> Model {
        Model {
            base: 0.0,
            row_bases: vec![0.0; rows],
            column_bases: vec![0.0; columns],
            row_features: vec![0.0; rows],
            column_features: vec![0.0; columns]
        }
    }
}
