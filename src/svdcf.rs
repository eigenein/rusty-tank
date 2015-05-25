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
    let mut input = helpers::get_input();
    let encyclopedia = encyclopedia::Encyclopedia::new();
    let (train_table, test_table) = helpers::read_stats(&mut input, MIN_BATTLES, &encyclopedia);
    println!("Initializing model.");
    let mut model = svd::Model::new(train_table.row_count(), encyclopedia.len(), FEATURE_COUNT);
    println!("Initial evaluation.");
    let train_error = helpers::evaluate(&model, &train_table);
    println!("Train error: {0:.6}.", train_error);
    let test_error = helpers::evaluate(&model, &test_table);
    println!("Test error: {0:.6}.", test_error);
    train(&mut model, &train_table, &test_table);
    let error_distribution = helpers::evaluate_error_distribution(&model, &test_table);
    println!("Test error distribution:");
    println!("------------------------");
    helpers::print_error_distribution(error_distribution);
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
        let train_error = helpers::evaluate(model, &train_table);
        let test_error = helpers::evaluate(model, &test_table);
        let drmse = rmse - previous_rmse;
        println!(
            "#{0} | {1:.2} sec | E: {2:.6} | dE: {3:.6} | train error: {4:.6} | test error: {5:.6}",
            step, helpers::get_seconds(start_time) / (step as f32 + 1.0), rmse, -drmse, train_error, test_error,
        );
        if rmse.is_nan() || drmse.abs() < MIN_DRMSE {
            break;
        }
        previous_rmse = rmse;
    }

    println!("Training finished in {:.1}s.", helpers::get_seconds(start_time));
}
