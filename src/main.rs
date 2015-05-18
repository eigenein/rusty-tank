//! # Rusty Tank
//!
//! Collaboration filtering for Tankopoisk.

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

/// Average tank rating.
const AVG_RATING: f64 = 0.51;
/// Maximum deviation of tank rating from `AVG_RATING`.
const MAX_DEVIATION: f64 = 0.1;
/// SVD feature count.
const FEATURE_COUNT: usize = 4;
/// Learning rate.
const RATE: f64 = 0.1;
/// Regularization parameter.
const LAMBDA: f64 = 8.0;

#[allow(dead_code)]
fn main() {
    let mut input = get_input();
    let encyclopedia = encyclopedia::Encyclopedia::new();
    let (train_table, test_table) = read_stats(&mut input, &encyclopedia);
    let mut model = svd::Model::new(train_table.row_count(), encyclopedia.len(), FEATURE_COUNT);
    println!("Initial evaluation.");
    let (train_avg, train_max) = evaluate(&model, &train_table);
    println!("Train | avg {:.6} | max {:.6}.", train_avg, train_max);
    let (test_avg, test_max) = evaluate(&model, &test_table);
    println!("Test  | avg {:.6} | max {:.6}.", test_avg, test_max);
    train(&mut model, &train_table, &test_table);
}

/// Reads statistics file.
///
/// Returns train rating table and test rating table.
fn read_stats<R: Read>(input: &mut R, encyclopedia: &encyclopedia::Encyclopedia) -> (csr::Csr, csr::Csr) {
    use rand::{Rng, thread_rng};

    use time::now;

    let start_time = time::now();
    let mut rng = rand::thread_rng();

    let mut train_table = csr::Csr::new();
    let mut test_table = csr::Csr::new();

    let mut non_empty_account_count = 0;
    let mut tank_battle_count = 0;

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

        let mut non_empty = false;

        match stats::read_account(input) {
            Some(account) => {
                for tank in account.tanks {
                    let tank_rating = tank.wins as f64 / tank.battles as f64;
                    if (tank_rating - AVG_RATING).abs() > MAX_DEVIATION {
                        continue;
                    }
                    non_empty = true;
                    tank_battle_count += tank.battles;
                    (if !rng.gen_weighted_bool(4) {
                        &mut train_table
                    } else {
                        &mut test_table
                    }).next(encyclopedia.get_column(tank.id), tank_rating);
                }
            }
            None => break
        }

        if non_empty {
            non_empty_account_count += 1;
        }
    }

    println!("Read {1} train and {2} test values in {0:.1}s.", get_seconds(start_time), train_table.len(), test_table.len());
    println!("Non-empty accounts: {0} ({1:.2}%).",
        non_empty_account_count, 100.0 * non_empty_account_count as f32 / train_table.row_count() as f32);
    println!("Average tank battles: {0:.2}.", tank_battle_count as f32 / (train_table.len() + test_table.len()) as f32);

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
    for step in 0.. {
        let rmse = model.make_step(RATE, LAMBDA, train_table);
        let (train_avg, train_max) = evaluate(model, &train_table);
        let (test_avg, test_max) = evaluate(model, &test_table);
        let drmse = rmse - previous_rmse;
        println!(
            "#{0} | {3:.3} sec | RMSE: {1:.6} | dE: {2:.7} | train: {4:.6} ({5:.6}) | test: {6:.6} ({7:.6})",
            step, rmse, drmse, get_seconds(start_time) / (step as f32 + 1.0),
            train_avg, train_max, test_avg, test_max,
        );
        if drmse.abs() < 0.0000001 {
            break;
        }
        previous_rmse = rmse;
    }

    println!("Training finished in {:.1}s.", get_seconds(start_time));
}

/// Evaluates the model.
fn evaluate(model: &svd::Model, table: &csr::Csr) -> (f64, f64) {
    let mut error_sum = 0.0;
    let mut error_max: f64 = 0.0;

    for row_index in 0..table.row_count() {
        for actual_value in table.get_row(row_index) {
            let error = (actual_value.value - model.predict(row_index, actual_value.column)).abs();
            error_sum += error;
            error_max = error_max.max(error);
        }
    }

    (error_sum / table.len() as f64, error_max)
}

/// Gets seconds elapsed since the specified time.
fn get_seconds(start_time: Tm) -> f32 {
    use time::now;

    (now() - start_time).num_milliseconds() as f32 / 1000.0
}
