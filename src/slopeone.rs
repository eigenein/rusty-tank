//! Slope One for Tankopoisk.

use std::collections::HashMap;

extern crate rand;
extern crate time;

mod csr;
mod encyclopedia;
mod helpers;
mod protobuf;
mod stats;

const MIN_BATTLES: u32 = 10;

struct Model {
    average_difference: HashMap<(u16, u16), f64>,
    rating_count: Vec<usize>,
}

impl Model {
    pub fn new(column_count: usize) -> Model {
        Model { average_difference: HashMap::new(), rating_count: vec![0; column_count] }
    }

    pub fn train(&mut self, matrix: &csr::Csr) {
        //
    }
}

impl helpers::AbstractModel for Model {
    fn predict(&self, train_matrix: &csr::Csr, row_index: usize, column_index: usize) -> f64 {
        0.0
    }
}

#[allow(dead_code)]
fn main() {
    let (encyclopedia, train_matrix, test_matrix) = helpers::get_stats(MIN_BATTLES, helpers::identity);
    println!("Training.");
    let mut model = Model::new(encyclopedia.len());
    model.train(&train_matrix);
    println!("Evaluating.");
    let train_error = helpers::evaluate(&model, &train_matrix, &train_matrix, helpers::identity);
    println!("Train error: {0:.6}.", train_error);
    let test_error = helpers::evaluate(&model, &train_matrix, &test_matrix, helpers::identity);
    println!("Test error: {0:.6}.", test_error);
    let error_distribution = helpers::evaluate_error_distribution(&model, &train_matrix, &test_matrix, helpers::identity);
    println!("Test error distribution:");
    println!("------------------------");
    helpers::print_error_distribution(error_distribution);
}
