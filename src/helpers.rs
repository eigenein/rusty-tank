//! Helper functions.

use std;
use std::fs::File;
use std::io::{BufReader, Read};

use time;

use csr;
use encyclopedia;
use stats;

pub const MAX_RATING: f64 = 100.0;

pub trait AbstractModel {
    /// Predicts value at the specified position.
    fn predict(&self, train_matrix: &csr::Csr, row_index: usize, column_index: usize) -> Option<f64>;
}

pub fn get_stats<F>(min_battles: u32, f: F) -> (encyclopedia::Encyclopedia, csr::Csr, csr::Csr)
    where F : Fn(f64) -> f64 {

    let mut input = get_input();
    let encyclopedia = encyclopedia::Encyclopedia::new();
    let (train_matrix, test_matrix) = read_stats(&mut input, min_battles, &encyclopedia, f);
    (encyclopedia, train_matrix, test_matrix)
}

/// Identity function. Returns the given value.
#[allow(dead_code)]
pub fn identity(value: f64) -> f64 {
    value
}

/// Gets seconds elapsed since the specified time.
pub fn get_seconds(start_time: time::Tm) -> f32 {
    use time::now;

    (now() - start_time).num_milliseconds() as f32 / 1000.0
}

/// Evaluates the model.
pub fn evaluate<F>(model: &AbstractModel, train_matrix: &csr::Csr, test_matrix: &csr::Csr, inverse_f: F) -> f64
    where F : Fn(f64) -> f64 {

    let mut error_count = 0;
    let mut error_sum = 0.0;

    for row_index in 0..test_matrix.row_count() {
        for actual_value in test_matrix.get_row(row_index) {
            if let Some(predicted_value) = model.predict(train_matrix, row_index, actual_value.column) {
                assert!(!predicted_value.is_nan());
                error_count += 1;
                error_sum += (inverse_f(predicted_value) - inverse_f(actual_value.value)).abs();
            }
        }
    }

    assert!(error_count != 0);
    error_sum / error_count as f64
}

/// Evaluates model error distribution.
pub fn evaluate_error_distribution<F>(model: &AbstractModel, train_matrix: &csr::Csr, test_matrix: &csr::Csr, inverse_f: F) -> Vec<f64>
    where F : Fn(f64) -> f64 {

    let mut distribution = vec![0.0; 102];
    let increment = 1.0 / test_matrix.len() as f64;

    for row_index in 0..test_matrix.row_count() {
        for actual_value in test_matrix.get_row(row_index) {
            if let Some(predicted_value) = model.predict(train_matrix, row_index, actual_value.column) {
                let error = inverse_f(predicted_value) - inverse_f(actual_value.value);
                distribution[error.abs().min(101.0).round() as usize] += increment;
            }
        }
    }

    distribution
}

/// Prints error distribution.
pub fn print_error_distribution(distribution: Vec<f64>) {
    let mut cumulative_frequency = 0.0;

    for (error, &frequency) in distribution.iter().enumerate() {
        cumulative_frequency += frequency;
        if frequency > 0.0001 {
            let bar = std::iter::repeat("x").take((500.0 * frequency) as usize).collect::<String>();
            println!("  {0:3}%: {1:.2}% {2}", error, 100.0 * cumulative_frequency, bar);
        }
    }
}

/// Gets statistics input.
fn get_input() -> BufReader<File> {
    use std::env::args;
    use std::path::Path;

    let input_file = File::open(&Path::new(&args().nth(1).unwrap())).unwrap();
    BufReader::with_capacity(1024 * 1024, input_file)
}

/// Reads statistics file.
///
/// Returns train rating matrix and test rating matrix.
fn read_stats<R: Read, F>(input: &mut R, min_battles: u32, encyclopedia: &encyclopedia::Encyclopedia, f: F) -> (csr::Csr, csr::Csr)
    where F : Fn(f64) -> f64 {

    use rand::{Rng, thread_rng};
    use time::now;

    let start_time = time::now();
    let mut rng = thread_rng();

    let mut train_matrix = csr::Csr::new();
    let mut test_matrix = csr::Csr::new();

    println!("Reading started at {}.", start_time.ctime());

    for i in 1.. {
        if i % 100000 == 0 {
            println!(
                "Reading | acc.: {} | {:.1} acc/s | train: {} | test: {}",
                i, i as f32 / get_seconds(start_time), train_matrix.len(), test_matrix.len()
            );
        }

        train_matrix.start();
        test_matrix.start();

        match stats::read_account(input) {
            Some(account) => {
                for tank in account.tanks {
                    if tank.battles < min_battles {
                        continue;
                    }
                    if tank.wins > tank.battles {
                        continue; // work around the bug in kit.py
                    }
                    (if !rng.gen_weighted_bool(20) {
                        &mut train_matrix
                    } else {
                        &mut test_matrix
                    }).next(encyclopedia.get_column(tank.id), f(MAX_RATING * tank.wins as f64 / tank.battles as f64));
                }
            }
            None => break
        }
    }

    println!(
        "Read {1} train and {2} test values in {0:.1}s. {3} rows.",
        get_seconds(start_time), train_matrix.len(), test_matrix.len(), train_matrix.row_count()
    );

    (train_matrix, test_matrix)
}
