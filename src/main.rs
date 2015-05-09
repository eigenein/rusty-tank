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

const MIN_BATTLES: u32 = 0;
const FEATURE_COUNT: usize = 4;
const STEP_COUNT: usize = 10;
const RATE: f32 = 0.5;
const LAMBDA: f32 = 0.0;

#[allow(dead_code)]
fn main() {
    let mut input = get_input();
    let encyclopedia = encyclopedia::Encyclopedia::new();
    let (train_table, test_table) = read_stats(&mut input, &encyclopedia);
    train(&train_table, &encyclopedia);
}

/// Reads statistics file.
///
/// Returns train rating table and test rating table.
fn read_stats<R: Read>(input: &mut R, encyclopedia: &encyclopedia::Encyclopedia) -> (csr::Csr, csr::Csr) {
    use time::now;

    use rand::{Rng, thread_rng};

    let start_time = time::now();
    let mut rng = rand::thread_rng();

    let mut train_table = csr::Csr::new();
    let mut test_table = csr::Csr::new();

    println!("Reading started at {}.", start_time.ctime());

    for i in 1.. {
        if i % 100000 == 0 {
            println!(
                "Reading | acc.: {} | {} acc/s | train: {} | test: {}",
                i, i / get_seconds(start_time), train_table.len(), test_table.len()
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
                    let rating = tank.wins as f32 / tank.battles as f32;
                    (if !rng.gen_weighted_bool(3) {
                        &mut train_table
                    } else {
                        &mut test_table
                    }).next(encyclopedia.get_column(tank.id), rating);
                }
            }
            None => break
        }
    }

    println!("Read {1} train and {2} test values in {0}s.", get_seconds(start_time), train_table.len(), test_table.len());

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
fn train(train_table: &csr::Csr, encyclopedia: &encyclopedia::Encyclopedia) {
    use std::f32;
    use time::now;

    let start_time = now();
    let mut model = svd::Model::new(train_table.row_count(), encyclopedia.len(), FEATURE_COUNT);

    println!("Training started at {}.", start_time.ctime());

    let mut previous_rmse = f32::INFINITY;
    for step in 0..STEP_COUNT {
        let rmse = model.make_step(RATE, LAMBDA, train_table);
        println!(
            "Training | step: {0} | {3} s/step | RMSE: {1:.6} | dRMSE: {2:.6}",
            step, rmse, rmse - previous_rmse, get_seconds(start_time) / (step as i64 + 1)
        );
        previous_rmse = rmse;
    }

    println!("Training finished in {}s.", get_seconds(start_time));
}

/// Gets seconds elapsed since the specified time.
fn get_seconds(start_time: Tm) -> i64 {
    use time::now;

    (now() - start_time).num_seconds()
}
