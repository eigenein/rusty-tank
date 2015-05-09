//! SVD implementation.
//!
//! See http://habrahabr.ru/company/surfingbird/blog/141959/.

use csr::{Csr, ColumnValue};

#[derive(Debug)]
pub struct Model {
    /// Base predictor.
    pub base: f32,
    /// Base row predictors.
    row_bases: Vec<f32>,
    /// Base column predictors.
    column_bases: Vec<f32>,
    /// Feature count.
    feature_count: usize,
    /// Row features.
    row_features: Vec<Vec<f32>>,
    /// Column features.
    column_features: Vec<Vec<f32>>,
}

impl Model {
    /// Creates a new model.
    pub fn new(row_count: usize, column_count: usize, feature_count: usize) -> Self {
        Model {
            base: 0.0,
            row_bases: vec![0.0; row_count],
            column_bases: vec![0.0; column_count],
            feature_count: feature_count,
            row_features: Model::new_feature_vectors(row_count, feature_count),
            column_features: Model::new_feature_vectors(column_count, feature_count),
        }
    }

    /// Makes a step.
    ///
    /// Returns RMSE.
    pub fn make_step(&mut self, rate: f32, lambda: f32, csr: &Csr) -> f32 {
        let mut rmse = 0.0;
        for row_index in 0..(csr.row_count() - 1) {
            let row = csr.get_row(row_index);
            for column_value in row {
                rmse += self.train(rate, lambda, row_index, column_value.column, column_value.value);
            }
        }
        (rmse / csr.len() as f32).sqrt()
    }

    /// Predicts value at the specified position.
    pub fn predict(&self, row_index: usize, column_index: usize) -> f32 {
        self.base + self.row_bases[row_index] + self.column_bases[column_index] + self.dot(row_index, column_index)
    }

    /// Creates a vector of feature vectors.
    fn new_feature_vectors(count: usize, feature_count: usize) -> Vec<Vec<f32>> {
        (0..count).map(|_| vec![0.0; feature_count]).collect()
    }

    /// Trains the model with the given sample.
    ///
    /// Returns squared error.
    fn train(&mut self, rate: f32, lambda: f32, row_index: usize, column_index: usize, value: f32) -> f32 {
        let error = value - self.predict(row_index, column_index);
        // Update baseline predictors.
        self.base += rate * error;
        self.row_bases[row_index] += rate * (error - lambda * self.row_bases[row_index]);
        self.column_bases[column_index] += rate * (error - lambda * self.column_bases[column_index]);
        // Update feature vectors.
        for i in 0..self.feature_count {
            self.row_features[row_index][i] += rate * (
                error * self.column_features[column_index][i] - lambda * self.row_features[row_index][i]);
            self.column_features[column_index][i] += rate * (
                error * self.row_features[row_index][i] - lambda * self.column_features[column_index][i]);
        }
        // Return squared error.
        error * error
    }

    /// Gets feature vectors dot product.
    fn dot(&self, row_index: usize, column_index: usize) -> f32 {
        (0..self.feature_count).fold(0.0, |acc, i| acc + self.row_features[row_index][i] * self.column_features[column_index][i])
    }
}

#[test]
fn test_make_step() {
    // Build a matrix.
    let mut csr = Csr::new();
    csr.start();
    csr.next(0, 1.0);
    csr.next(1, 2.0);
    csr.start();
    csr.next(0, 3.0);
    csr.next(2, 4.0);
    csr.start();
    csr.next(1, 5.0);
    csr.next(2, 6.0);
    csr.start();
    // Build a model.
    let mut model = Model::new(3, 3, 1);
    // Train the model.
    const rate: f32 = 0.001;
    const lambda: f32 = 1.0;

    let mut previous_rmse = model.make_step(rate, lambda, &csr);
    for _ in 0..100 {
        let rmse = model.make_step(rate, lambda, &csr);
        assert!(rmse < previous_rmse);
        previous_rmse = rmse;
    }
}
