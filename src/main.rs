//! # Rusty Tank
//!
//! Collaboration filtering for Tankopoisk.

use std::fs::File;
use std::io::{BufReader, Read};
use std::collections::HashMap;

extern crate rand;
extern crate time;

use time::Tm;

mod csr;
mod encyclopedia;
mod protobuf;
mod stats;
mod svd;

/// Minimum battle count to train with.
const MIN_BATTLES: u32 = 10;
/// SVD feature count.
const FEATURE_COUNT: usize = 2;
/// Learning rate.
const RATE: f64 = 0.1;
/// Regularization parameter.
const LAMBDA: f64 = 2.0;

#[allow(dead_code)]
fn main() {
    let mut input = get_input();
    let encyclopedia = encyclopedia::Encyclopedia::new();
    let (train_table, test_table, overall_rating) = read_stats(&mut input, &encyclopedia);
    let mut model = svd::Model::new(train_table.row_count(), encyclopedia.len(), FEATURE_COUNT);
    println!("Initial evaluation.");
    let initial_train_score = 100.0 * evaluate(&model, &train_table, &overall_rating);
    println!("Train score: {0:.2}.", initial_train_score);
    let initial_test_score = 100.0 * evaluate(&model, &test_table, &overall_rating);
    println!("Test score: {0:.2}.", initial_test_score);
    train(&mut model, &train_table, &test_table, &overall_rating);
}

/// Reads statistics file.
///
/// Returns train rating table and test rating table.
fn read_stats<R: Read>(input: &mut R, encyclopedia: &encyclopedia::Encyclopedia) -> (csr::Csr, csr::Csr, HashMap<usize, f64>) {
    use rand::{Rng, thread_rng};

    use time::now;

    let start_time = time::now();
    let mut rng = rand::thread_rng();

    let mut train_table = csr::Csr::new();
    let mut test_table = csr::Csr::new();
    let mut overall_rating = HashMap::new();

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
                let mut account_wins = 0;
                let mut account_battles = 0;
                for tank in account.tanks {
                    if tank.battles < MIN_BATTLES {
                        continue;
                    }
                    let rating = tank.wins as f64 / tank.battles as f64;
                    (if !rng.gen_weighted_bool(3) {
                    let tank_rating = tank.wins as f64 / tank.battles as f64;
                    (if !rng.gen_weighted_bool(4) {
                        account_wins += tank.wins;
                        account_battles += tank.battles;
                        &mut train_table
                    } else {
                        &mut test_table
                    }).next(encyclopedia.get_column(tank.id), tank_rating);
                }
                if account_battles != 0 {
                    overall_rating.insert(train_table.row_count(), account_wins as f64 / account_battles as f64);
                }
            }
            None => break
        }
    }

    println!("Read {1} train and {2} test values in {0:.1}s.", get_seconds(start_time), train_table.len(), test_table.len());

    (train_table, test_table, overall_rating)
}

/// Gets statistics input.
fn get_input() -> BufReader<File> {
    use std::env::args;
    use std::path::Path;

    let input_file = File::open(&Path::new(&args().nth(1).unwrap())).unwrap();
    BufReader::with_capacity(1024 * 1024, input_file)
}

/// Trains the model.
fn train(model: &mut svd::Model, train_table: &csr::Csr, test_table: &csr::Csr, overall_rating: &HashMap<usize, f64>) {
    use std::f64;
    use time::now;

    let start_time = now();

    println!("Training started at {}.", start_time.ctime());

    let mut previous_rmse = f64::INFINITY;
    for step in 0.. {
        let rmse = model.make_step(RATE, LAMBDA, train_table);
        let train_score = 100.0 * evaluate(model, &train_table, &overall_rating);
        let test_score = 100.0 * evaluate(model, &test_table, &overall_rating);
        let drmse = rmse - previous_rmse;
        println!(
            "#{0} | {3:.3} sec | RMSE: {1:.6} | dE: {2:.7} | train: {4:.3} | test: {5:.3}",
            step, rmse, drmse, get_seconds(start_time) / (step as f32 + 1.0), train_score, test_score,
        );
        if drmse.abs() < 0.000001 {
            break;
        }
        previous_rmse = rmse;
    }

    println!("Training finished in {:.1}s.", get_seconds(start_time));
}

/// Evaluates the model.
fn evaluate(model: &svd::Model, table: &csr::Csr, overall_rating: &HashMap<usize, f64>) -> f64 {
    let mut value_count = 0;
    let mut true_count = 0;
    for row_index in 0..(table.row_count() - 1) {
        let account_rating = match overall_rating.get(&row_index) {
            Some(&account_rating) => account_rating,
            None => continue,
        };
        for actual_value in table.get_row(row_index) {
            value_count += 1;
            let predicted_value = model.predict(row_index, actual_value.column);
            if (predicted_value > account_rating) == (actual_value.value > account_rating) {
                true_count += 1;
            }
        }
    }
    true_count as f64 / value_count as f64
}

/// Gets seconds elapsed since the specified time.
fn get_seconds(start_time: Tm) -> f32 {
    use time::now;

    (now() - start_time).num_milliseconds() as f32 / 1000.0
}
