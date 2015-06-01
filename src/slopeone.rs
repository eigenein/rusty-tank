//! Slope One for Tankopoisk.

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
    average_differences: Vec<Option<f64>>,
    rating_count: Vec<usize>,
}

impl Model {
    pub fn new(column_count: usize) -> Model {
        Model { column_count: column_count, average_differences: Vec::new(), rating_count: Vec::new() }
    }

    pub fn train(&mut self, matrix: &csr::Csr) {
        let mut difference_sums = vec![0.0f64; self.column_count * self.column_count];
        let mut difference_count = vec![0; self.column_count * self.column_count];
        // Reset model.
        self.average_differences = vec![None; self.column_count * self.column_count];
        self.rating_count = vec![0; self.column_count];
        // Calculate differences.
        for row_index in 0..matrix.row_count() {
            let row = matrix.get_row(row_index);
            for value_1 in row {
                self.rating_count[value_1.column] += 1;
                for value_2 in row {
                    if value_1.column != value_2.column {
                        let index = self.flat_index(value_1.column, value_2.column);
                        difference_sums[index] += value_1.value - value_2.value;
                        difference_count[index] += 1;
                    }
                }
            }
        }
        // Average differences.
        for column_1 in 0..(self.column_count - 1) {
            for column_2 in 0..(self.column_count - 1) {
                if column_1 != column_2 {
                    let index = self.flat_index(column_1, column_2);
                    if difference_count[index] != 0 {
                        self.average_differences[index] = Some(difference_sums[index] / difference_count[index] as f64);
                    }
                }
            }
        }
    }

    /// Flats the pair of indexes.
    fn flat_index(&self, column_1: usize, column_2: usize) -> usize {
        column_1 * self.column_count + column_2
    }
}

impl helpers::AbstractModel for Model {
    fn predict(&self, train_matrix: &csr::Csr, row_index: usize, column_index: usize) -> Option<f64> {
        let mut sum = 0.0f64;
        let mut weight = 0;

        for value in train_matrix.get_row(row_index) {
            if value.column != column_index {
                if let Some(diff) = self.average_differences[self.flat_index(column_index, value.column)] {
                    sum += self.rating_count[value.column] as f64 * (value.value + diff);
                    weight += self.rating_count[value.column];
                }
            }
        }

        if weight != 0 { Some(sum / weight as f64) } else { None }
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
