//! SVD-based collaboration filtering for Tankopoisk.

extern crate rand;
extern crate time;

mod csr;
mod encyclopedia;
mod helpers;
mod protobuf;
mod stats;
mod svd;

/// Minimum battles count.
const MIN_BATTLES: u32 = 10;
/// SVD feature count.
const FEATURE_COUNT: usize = 4;
/// Learning rate.
const RATE: f64 = 0.001;
/// Regularization parameter.
const LAMBDA: f64 = 16.0;
/// Minimum train RMSE change.
const MIN_DRMSE: f64 = 0.000001;
/// Maximum train iteration count.
const MAX_ITERATION_COUNT: usize = 500;

#[allow(dead_code)]
fn main() {
    let (encyclopedia, train_matrix, test_matrix) = helpers::get_stats(MIN_BATTLES, helpers::identity);
    println!("Initializing model.");
    let mut model = svd::Model::new(train_matrix.row_count(), encyclopedia.len(), FEATURE_COUNT);
    println!("Initial evaluation.");
    let train_error = helpers::evaluate(&model, &train_matrix, &train_matrix, helpers::identity);
    println!("Train error: {0:.6}.", train_error);
    let test_error = helpers::evaluate(&model, &train_matrix, &test_matrix, helpers::identity);
    println!("Test error: {0:.6}.", test_error);
    train(&mut model, &train_matrix, &test_matrix);
    let error_distribution = helpers::evaluate_error_distribution(&model, &train_matrix, &test_matrix, helpers::identity);
    println!("Test error distribution:");
    println!("------------------------");
    helpers::print_error_distribution(error_distribution);
}

/// Trains the model.
fn train(model: &mut svd::Model, train_matrix: &csr::Csr, test_matrix: &csr::Csr) {
    use std::f64;
    use time::now;

    let start_time = now();

    println!("Training started at {}.", start_time.ctime());

    let mut previous_rmse = f64::INFINITY;
    for step in 0..MAX_ITERATION_COUNT {
        let rmse = model.make_step(RATE, LAMBDA, train_matrix);
        let train_error = helpers::evaluate(model, &train_matrix, &train_matrix, helpers::identity);
        let test_error = helpers::evaluate(model, &train_matrix, &test_matrix, helpers::identity);
        let drmse = rmse - previous_rmse;
        println!(
            "#{0} | {1:.2} sec | E: {2:.6} | dE: {3:.6} | train error: {4:.6} | test error: {5:.6}",
            step, helpers::get_seconds(start_time) / (step as f32 + 1.0), rmse, -drmse, train_error, test_error,
        );
        if rmse.is_nan() || drmse.abs() < MIN_DRMSE || drmse > 0.0 {
            break;
        }
        previous_rmse = rmse;
    }

    println!("Training finished in {:.1}s.", helpers::get_seconds(start_time));
}

const F_SCALE: f64 = 4.0;

#[allow(dead_code)]
fn sigmoid(value: f64) -> f64 {
    100.0 / (1.0 + ((50.0 - value) / F_SCALE).exp())
}

#[allow(dead_code)]
fn inverse_sigmoid(value: f64) -> f64 {
    50.0 - F_SCALE * (100.0 / value - 1.0).ln()
}
