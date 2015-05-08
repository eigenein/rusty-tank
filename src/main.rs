//! # Rusty Tank
//!
//! Collaboration filtering for Tankopoisk.

use std::fs::File;
use std::io::{BufReader, Read};

extern crate rand;
extern crate time;

mod csr;
mod encyclopedia;
mod protobuf;
mod stats;
mod svd;

#[allow(dead_code)]
fn main() {
    let mut input = get_input();
    let (train_table, test_table) = read_stats(&mut input, encyclopedia::Encyclopedia::new());
}

/// Reads statistics file.
///
/// Returns train rating table and test rating table.
fn read_stats<R: Read>(input: &mut R, encyclopedia: encyclopedia::Encyclopedia) -> (csr::Csr, csr::Csr) {
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
                i, i / (now() - start_time).num_seconds(), train_table.len(), test_table.len()
            );
        }

        train_table.start();
        test_table.start();

        match stats::read_account(input) {
            Some(account) => {
                for tank in account.tanks {
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

    println!(
        "Read {1} train and {2} test values in {0}s.",
        (now() - start_time).num_seconds(), train_table.len(), test_table.len()
    );

    (train_table, test_table)
}

/// Gets statistics input.
fn get_input() -> BufReader<File> {
    use std::env::args;
    use std::path::Path;

    let input_file = File::open(&Path::new(&args().nth(1).unwrap())).unwrap();
    BufReader::with_capacity(1024 * 1024, input_file)
}
