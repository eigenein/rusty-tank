//! SVD-based collaboration filtering for Tankopoisk.

use std::fs::File;
use std::io::{BufReader, Read};

extern crate rand;
extern crate time;

use time::Tm;

mod csr;
mod encyclopedia;
mod protobuf;
mod stats;
mod svd;

/// Minimum battles count.
const MIN_BATTLES: u32 = 10;
/// SVD feature count.
const FEATURE_COUNT: usize = 64;
/// Learning rate.
const RATE: f64 = 0.001;
/// Regularization parameter.
const LAMBDA: f64 = 1.0;
/// Minimum train RMSE change.
const MIN_DRMSE: f64 = 0.000001;
/// Maximum train iteration count.
const MAX_ITERATION_COUNT: usize = 500;

#[allow(dead_code)]
fn main() {
    let mut input = get_input();
    let encyclopedia = encyclopedia::Encyclopedia::new();
    let (train_table, test_table) = read_stats(&mut input, &encyclopedia);
    println!("Initializing model.");
    let mut model = svd::Model::new(train_table.row_count(), encyclopedia.len(), FEATURE_COUNT);
    println!("Initial evaluation.");
    let train_error = evaluate(&model, &train_table);
    println!("Train error: {0:.6}.", train_error);
    let test_error = evaluate(&model, &test_table);
    println!("Test error: {0:.6}.", test_error);
    train(&mut model, &train_table, &test_table);
    let error_distribution = evaluate_error_distribution(&model, &test_table);
    println!("Test error distribution:");
    println!("------------------------");
    print_error_distribution(error_distribution);
}

/// Reads statistics file.
///
/// Returns train rating table and test rating table.
fn read_stats<R: Read>(input: &mut R, encyclopedia: &encyclopedia::Encyclopedia) -> (csr::Csr, csr::Csr) {
    use rand::{Rng, thread_rng};
    use time::now;

    let start_time = time::now();
    let mut rng = thread_rng();

    let mut train_table = csr::Csr::new();
    let mut test_table = csr::Csr::new();

    println!("Reading started at {}.", start_time.ctime());

    for i in 1.. {
        if i % 100000 == 0 {
            println!(
                "Reading | acc.: {} | {:.1} acc/s | train: {} | test: {}",
                i, i as f32 / get_seconds(start_time), train_table.len(), test_table.len()
            );
        }

        train_table.start();
        test_table.start();

        match stats::read_account(input) {
            Some(account) => {
                for tank in account.tanks {
                    if tank.battles < MIN_BATTLES {
                        continue;
                    }
                    if tank.wins > tank.battles {
                        continue; // work around the bug in kit.py
                    }
                    (if !rng.gen_weighted_bool(20) {
                        &mut train_table
                    } else {
                        &mut test_table
                    }).next(encyclopedia.get_column(tank.id), 100.0 * tank.wins as f64 / tank.battles as f64);
                }
            }
            None => break
        }
    }

    println!("Read {1} train and {2} test values in {0:.1}s.", get_seconds(start_time), train_table.len(), test_table.len());

    (train_table, test_table)
}

/// Gets statistics input.
fn get_input() -> BufReader<File> {
    use std::env::args;
    use std::path::Path;

    let input_file = File::open(&Path::new(&args().nth(1).unwrap())).unwrap();
    BufReader::with_capacity(1024 * 1024, input_file)
}

/// Trains the model.
fn train(model: &mut svd::Model, train_table: &csr::Csr, test_table: &csr::Csr) {
    use std::f64;
    use time::now;

    let start_time = now();

    println!("Training started at {}.", start_time.ctime());

    let mut previous_rmse = f64::INFINITY;
    for step in 0..MAX_ITERATION_COUNT {
        let rmse = model.make_step(RATE, LAMBDA, train_table);
        let train_error = evaluate(model, &train_table);
        let test_error = evaluate(model, &test_table);
        let drmse = rmse - previous_rmse;
        println!(
            "#{0} | {1:.2} sec | E: {2:.6} | dE: {3:.6} | train error: {4:.6} | test error: {5:.6}",
            step, get_seconds(start_time) / (step as f32 + 1.0), rmse, -drmse, train_error, test_error,
        );
        if rmse.is_nan() || drmse.abs() < MIN_DRMSE {
            break;
        }
        previous_rmse = rmse;
    }

    println!("Training finished in {:.1}s.", get_seconds(start_time));
}

/// Evaluates the model.
fn evaluate(model: &svd::Model, table: &csr::Csr) -> f64 {
    let mut error = 0.0;

    for row_index in 0..table.row_count() {
        for actual_value in table.get_row(row_index) {
            error += (model.predict(row_index, actual_value.column) - actual_value.value).abs();
        }
    }

    error / table.len() as f64
}

/// Evaluates model error distribution.
fn evaluate_error_distribution(model: &svd::Model, table: &csr::Csr) -> Vec<f64> {
    let mut distribution = vec![0.0; 102];
    let increment = 1.0 / table.len() as f64;

    for row_index in 0..table.row_count() {
        for actual_value in table.get_row(row_index) {
            let error = (model.predict(row_index, actual_value.column) - actual_value.value).abs().min(101.0);
            distribution[error.round() as usize] += increment;
        }
    }

    distribution
}

/// Prints error distribution.
fn print_error_distribution(distribution: Vec<f64>) {
    let mut cumulative_frequency = 0.0;

    for (error, &frequency) in distribution.iter().enumerate() {
        cumulative_frequency += frequency;
        if frequency > 0.0001 {
            let bar = std::iter::repeat("x").take((1000.0 * frequency) as usize).collect::<String>();
            println!("  {0:3}%: {1:.2}% {2}", error, 100.0 * cumulative_frequency, bar);
        }
    }
}

/// Gets seconds elapsed since the specified time.
fn get_seconds(start_time: Tm) -> f32 {
    use time::now;

    (now() - start_time).num_milliseconds() as f32 / 1000.0
}
