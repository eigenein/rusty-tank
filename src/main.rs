//! # Rusty Tank
//!
//! Collaboration filtering for Tankopoisk.

use std::fs::File;
use std::io::{BufReader, Read};

extern crate time;

mod csr;
mod protobuf;
mod stats;
mod svd;

#[allow(dead_code)]
fn main() {
    let mut input = get_input();
    let storage = read_stats(&mut input);
}

/// Reads statistics file.
fn read_stats<R: Read>(input: &mut R) -> csr::CompressedStorage<u16, f32> {
    use time::now;

    let start_time = time::now();
    let mut last_progress_report = start_time;
    let mut storage = csr::CompressedStorage::new();
    println!("Reading started at {}.", start_time.ctime());

    for i in 1.. {
        if now().tm_sec != last_progress_report.tm_sec {
            println!("Reading | {} accounts | {} values", i, storage.len());
            last_progress_report = now();
        }
        match stats::read_account(input) {
            Some(account) => {
                storage.start();
                for tank in account.tanks {
                    storage.next(tank.id, tank.wins as f32 / tank.battles as f32);
                }
            }
            None => break
        }
    }

    storage.start();
    println!("Read {} values in {}s.", storage.len(), (now() - start_time).num_seconds());
    storage
}

/// Gets statistics input.
fn get_input() -> BufReader<File> {
    use std::env::args;
    use std::path::Path;

    let input_file = File::open(&Path::new(&args().nth(1).unwrap())).unwrap();
    BufReader::with_capacity(1024 * 1024, input_file)
}
