//! Helper functions.

use std::fs::File;
use std::io::{BufReader, Read};

use time;

use csr;
use encyclopedia;
use stats;

/// Gets statistics input.
pub fn get_input() -> BufReader<File> {
    use std::env::args;
    use std::path::Path;

    let input_file = File::open(&Path::new(&args().nth(1).unwrap())).unwrap();
    BufReader::with_capacity(1024 * 1024, input_file)
}

/// Reads statistics file.
///
/// Returns train rating table and test rating table.
pub fn read_stats<R: Read>(input: &mut R, min_battles: u32, encyclopedia: &encyclopedia::Encyclopedia) -> (csr::Csr, csr::Csr) {
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
                    if tank.battles < min_battles {
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

/// Gets seconds elapsed since the specified time.
pub fn get_seconds(start_time: time::Tm) -> f32 {
    use time::now;

    (now() - start_time).num_milliseconds() as f32 / 1000.0
}
