//! Naive rating prediction for Tankopoisk.

extern crate rand;
extern crate time;

mod csr;
mod encyclopedia;
mod helpers;
mod protobuf;
mod stats;

const MIN_BATTLES: u32 = 10;

struct Model {
    column_count: usize,
    average_rating: Vec<f64>,
}

impl Model {
    pub fn new(column_count: usize) -> Model {
        Model { column_count: column_count, average_rating: vec![0.0f64; column_count] }
    }

    pub fn train(&mut self, matrix: &csr::Csr) {
        let mut rating_count = vec![0; self.column_count];

        for row_index in 0..(matrix.row_count() - 1) {
            let row = matrix.get_row(row_index);
            for value in row {
                self.average_rating[value.column] += value.value;
                rating_count[value.column] += 1;
            }
        }

        for column_index in 0..(self.column_count - 1) {
            self.average_rating[column_index] /= rating_count[column_index] as f64;
        }
    }
}

impl helpers::AbstractModel for Model {
    #[allow(unused_variables)]
    fn predict(&self, row_index: usize, column_index: usize) -> f64 {
        self.average_rating[column_index]
    }
}

fn main() {
    let (encyclopedia, train_matrix, test_matrix) = helpers::get_stats(MIN_BATTLES, helpers::identity);
    println!("Training.");
    let mut model = Model::new(encyclopedia.len());
    model.train(&train_matrix);
    println!("Evaluating.");
    let train_error = helpers::evaluate(&model, &train_matrix, helpers::identity);
    println!("Train error: {0:.6}.", train_error);
    let test_error = helpers::evaluate(&model, &test_matrix, helpers::identity);
    println!("Test error: {0:.6}.", test_error);
    let error_distribution = helpers::evaluate_error_distribution(&model, &test_matrix, helpers::identity);
    println!("Test error distribution:");
    println!("------------------------");
    helpers::print_error_distribution(error_distribution);
}
