//! Item-based collaborative filtering.

extern crate rand;
extern crate time;

mod csr;
mod encyclopedia;
mod helpers;
mod protobuf;
mod stats;

const MIN_BATTLES: u32 = 10;

/// Collaborative filtering model.
struct Model {
    row_count: usize,
    /// Correlations between rows.
    correlations: Vec<f64>,
}

/// Gets Pearson correlation coefficient.
fn pearson(a: csr::Row, b: csr::Row) -> f64 {
    use std::collections::HashMap;
    // Map A column indexes into corresponding values.
    let mut a_map: HashMap<usize, f64> = HashMap::new();
    for column_value in a {
        a_map.insert(column_value.column, column_value.value);
    }
    // Get the sums.
    let mut n = 0;
    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    let mut sum_squared_a = 0.0;
    let mut sum_squared_b = 0.0;
    let mut product_sum = 0.0;
    for b_column_value in b {
        if let Some(&a_value) = a_map.get(&b_column_value.column) {
            n += 1;
            sum_a += a_value;
            sum_b += b_column_value.value;
            sum_squared_a += a_value * a_value;
            sum_squared_b += b_column_value.value * b_column_value.value;
            product_sum += a_value * b_column_value.value;
        }
    }
    // Get coefficient.
    if n == 0 {
        return 0.0;
    }
    let numerator = product_sum - (sum_a * sum_b / n as f64);
    let denominator = ((sum_squared_a - sum_a * sum_a / n as f64) * (sum_squared_b - sum_b * sum_b / n as f64)).sqrt();
    if denominator.abs() < 1e-9 {
        return 0.0;
    }
    numerator / denominator
}

#[allow(dead_code)]
fn main() {
    let (encyclopedia, train_matrix, test_matrix) = helpers::get_stats(MIN_BATTLES, helpers::identity);
    // TODO: transpose train_matrix.
}

#[test]
fn test_pearson() {
    let mut matrix = csr::Csr::new();

    const LADY_IN_THE_WATER: usize = 0;
    const SNAKES_ON_A_PLANE: usize = 1;
    const JUST_MY_LUCK: usize = 2;
    const SUPERMAN_RETURNS: usize = 3;
    const YOU_ME_AND_DUPREE: usize = 4;
    const THE_NIGHT_LISTENER: usize = 5;

    // Lisa Rose.
    matrix.start();
    matrix.next(LADY_IN_THE_WATER, 2.5);
    matrix.next(SNAKES_ON_A_PLANE, 3.5);
    matrix.next(JUST_MY_LUCK, 3.0);
    matrix.next(SUPERMAN_RETURNS, 3.5);
    matrix.next(YOU_ME_AND_DUPREE, 2.5);
    matrix.next(THE_NIGHT_LISTENER, 3.0);

    // Gene Seymour.
    matrix.start();
    matrix.next(LADY_IN_THE_WATER, 3.0);
    matrix.next(SNAKES_ON_A_PLANE, 3.5);
    matrix.next(JUST_MY_LUCK, 1.5);
    matrix.next(SUPERMAN_RETURNS, 5.0);
    matrix.next(YOU_ME_AND_DUPREE, 3.5);
    matrix.next(THE_NIGHT_LISTENER, 3.0);

    // Michael Phillips.
    matrix.start();
    matrix.next(LADY_IN_THE_WATER, 2.5);
    matrix.next(SNAKES_ON_A_PLANE, 3.0);
    matrix.next(SUPERMAN_RETURNS, 3.5);
    matrix.next(THE_NIGHT_LISTENER, 4.0);

    // Claudia Puig.
    matrix.start();
    matrix.next(SNAKES_ON_A_PLANE, 3.5);
    matrix.next(JUST_MY_LUCK, 3.0);
    matrix.next(SUPERMAN_RETURNS, 4.0);
    matrix.next(YOU_ME_AND_DUPREE, 2.5);
    matrix.next(THE_NIGHT_LISTENER, 4.5);

    // Mick LaSalle.
    matrix.start();
    matrix.next(LADY_IN_THE_WATER, 3.0);
    matrix.next(SNAKES_ON_A_PLANE, 4.0);
    matrix.next(JUST_MY_LUCK, 2.0);
    matrix.next(SUPERMAN_RETURNS, 3.0);
    matrix.next(YOU_ME_AND_DUPREE, 2.0);
    matrix.next(THE_NIGHT_LISTENER, 3.0);

    // Jack Matthews.
    matrix.start();
    matrix.next(LADY_IN_THE_WATER, 3.0);
    matrix.next(SNAKES_ON_A_PLANE, 4.0);
    matrix.next(SUPERMAN_RETURNS, 5.0);
    matrix.next(YOU_ME_AND_DUPREE, 3.5);
    matrix.next(THE_NIGHT_LISTENER, 3.0);

    // Toby.
    matrix.start();
    matrix.next(SNAKES_ON_A_PLANE, 4.5);
    matrix.next(SUPERMAN_RETURNS, 4.0);
    matrix.next(YOU_ME_AND_DUPREE, 1.0);

    // Unknown Artist.
    matrix.start();
    matrix.next(YOU_ME_AND_DUPREE, 4.5);

    matrix.start();

    assert_eq!(pearson(matrix.get_row(0), matrix.get_row(1)), 0.39605901719066976);
    assert_eq!(pearson(matrix.get_row(6), matrix.get_row(0)), 0.99124070716192991);
    assert_eq!(pearson(matrix.get_row(6), matrix.get_row(3)), 0.89340514744156474);
    assert_eq!(pearson(matrix.get_row(6), matrix.get_row(4)), 0.92447345164190486);
    assert_eq!(pearson(matrix.get_row(6), matrix.get_row(7)), 0.0);
}
