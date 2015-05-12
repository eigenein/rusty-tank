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
const MIN_BATTLES: u32 = 5;
/// SVD feature count.
const FEATURE_COUNT: usize = 2;
/// SVD step count,
const STEP_COUNT: usize = 100;
/// Learning rate.
const RATE: f32 = 0.1;
/// Learning rate multiplier.
const RATE_MULTIPLIER: f32 = 1.00;
/// Regularization parameter.
const LAMBDA: f32 = 0.0;

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
fn read_stats<R: Read>(input: &mut R, encyclopedia: &encyclopedia::Encyclopedia) -> (csr::Csr, csr::Csr, HashMap<usize, f32>) {
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
                for tank in &account.tanks {
                    if tank.battles >= MIN_BATTLES {
                        account_wins += tank.wins;
                        account_battles += tank.battles;
                    }
                }
                if account_battles == 0 {
                    continue;
                }
                let account_rating = account_wins as f32 / account_battles as f32;
                overall_rating.insert(train_table.row_count(), account_rating);
                for tank in &account.tanks {
                    if tank.battles >= MIN_BATTLES {
                        let tank_rating = tank.wins as f32 / tank.battles as f32;
                        (if !rng.gen_weighted_bool(3) {
                            &mut train_table
                        } else {
                            &mut test_table
                        }).next(encyclopedia.get_column(tank.id), if tank_rating < account_rating { -1.0 } else { 1.0 });
                    }
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
fn train(model: &mut svd::Model, train_table: &csr::Csr, test_table: &csr::Csr, overall_rating: &HashMap<usize, f32>) {
    use std::f32;
    use time::now;

    let start_time = now();
    let mut rate = RATE;

    println!("Training started at {}.", start_time.ctime());

    let mut previous_rmse = f32::INFINITY;
    for step in 0..STEP_COUNT {
        let rmse = model.make_step(rate, LAMBDA, train_table);
        let train_score = 100.0 * evaluate(model, &train_table, &overall_rating);
        let test_score = 100.0 * evaluate(model, &test_table, &overall_rating);
        println!(
            "#{0} | {3:.2} sec | rate: {6:.3} | RMSE: {1:.6} | dE: {2:.6} | train: {4:.2} | test: {5:.2}",
            step, rmse, rmse - previous_rmse, get_seconds(start_time) / (step as f32 + 1.0), train_score, test_score, rate,
        );
        if rmse < previous_rmse {
            rate *= RATE_MULTIPLIER;
        } else {
            rate /= RATE_MULTIPLIER * RATE_MULTIPLIER * RATE_MULTIPLIER;
        }
        previous_rmse = rmse;
    }

    println!("Training finished in {:.1}s.", get_seconds(start_time));
}

/// Evaluates the model.
fn evaluate(model: &svd::Model, table: &csr::Csr, overall_rating: &HashMap<usize, f32>) -> f32 {
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
            if (predicted_value > 0.0) == (actual_value.value > account_rating) {
                true_count += 1;
            }
        }
    }
    true_count as f32 / value_count as f32
}

/// Gets seconds elapsed since the specified time.
fn get_seconds(start_time: Tm) -> f32 {
    use time::now;

    (now() - start_time).num_milliseconds() as f32 / 1000.0
}
