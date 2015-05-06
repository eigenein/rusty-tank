//! # Rusty Tank
//!
//! Collaboration filtering for Tankopoisk.

use std::fs::File;
use std::io::{BufReader, Read};

extern crate time;

mod protobuf;
mod stats;
mod svd;

fn main() {
    let mut input = get_input();
    read_stats(&mut input);
}

/// Reads statistics file.
fn read_stats<R: Read>(input: &mut R) {
    use time::now;

    let start_time = time::now();
    let mut last_progress_report = start_time;
    println!("Reading started at {}.", start_time.ctime());

    for i in 1.. {
        if now().tm_sec != last_progress_report.tm_sec {
            println!("Reading | {} accounts", i);
            last_progress_report = now();
        }
        match stats::read_account(input) {
            Some(account) => {
                // TODO.
            }
            None => break
        }
    }

    println!("Read in {}s.", (now() - start_time).num_seconds());
}

/// Gets statistics input.
fn get_input() -> BufReader<File> {
    use std::env::args;
    use std::path::Path;

    let input_file = File::open(&Path::new(&args().nth(1).unwrap())).unwrap();
    BufReader::with_capacity(1024 * 1024, input_file)
}
