//! Correlation functions.

use csr;

/// Pearson correlation.
fn pearson(a: csr::Row, b: csr::Row) -> f64 {
    let mut n = 0;

    if n != 0 {
        //
    } else {
        0.0
    }
}

#[test]
fn test_pearson() {
    let mut table = csr::Csr::new();

    const LADY_IN_THE_WATER: usize = 0;
    const SNAKES_ON_A_PLANE: usize = 1;
    const JUST_MY_LUCK: usize = 2;
    const SUPERMAN_RETURNS: usize = 3;
    const YOU_ME_AND_DUPREE: usize = 4;
    const THE_NIGHT_LISTENER: usize = 5;

    // Lisa Rose.
    table.start();
    table.next(LADY_IN_THE_WATER, 2.5);
    table.next(SNAKES_ON_A_PLANE, 3.5);
    table.next(JUST_MY_LUCK, 3.0);
    table.next(SUPERMAN_RETURNS, 3.5);
    table.next(YOU_ME_AND_DUPREE, 2.5);
    table.next(THE_NIGHT_LISTENER, 3.0);

    // Gene Seymour.
    table.start();
    table.next(LADY_IN_THE_WATER, 3.0);
    table.next(SNAKES_ON_A_PLANE, 3.5);
    table.next(JUST_MY_LUCK, 1.5);
    table.next(SUPERMAN_RETURNS, 5.0);
    table.next(YOU_ME_AND_DUPREE, 3.5);
    table.next(THE_NIGHT_LISTENER, 3.0);

    // Michael Phillips.
    table.next(LADY_IN_THE_WATER, 2.5);
    table.next(SNAKES_ON_A_PLANE, 3.0);
    table.next(SUPERMAN_RETURNS, 3.5);
    table.next(THE_NIGHT_LISTENER, 4.0);

    // Claudia Puig.
    table.next(SNAKES_ON_A_PLANE, 3.5);
    table.next(JUST_MY_LUCK, 3.0);
    table.next(SUPERMAN_RETURNS, 4.0);
    table.next(YOU_ME_AND_DUPREE, 2.5);
    table.next(THE_NIGHT_LISTENER, 4.5);

    // Mick LaSalle.
    table.next(LADY_IN_THE_WATER, 3.0);
    table.next(SNAKES_ON_A_PLANE, 4.0);
    table.next(JUST_MY_LUCK, 2.0);
    table.next(SUPERMAN_RETURNS, 3.0);
    table.next(YOU_ME_AND_DUPREE, 2.0);
    table.next(THE_NIGHT_LISTENER, 3.0);

    // Jack Matthews.
    table.next(LADY_IN_THE_WATER, 3.0);
    table.next(SNAKES_ON_A_PLANE, 4.0);
    table.next(SUPERMAN_RETURNS, 5.0);
    table.next(YOU_ME_AND_DUPREE, 3.5);
    table.next(THE_NIGHT_LISTENER, 3.0);

    // Toby.
    table.next(SNAKES_ON_A_PLANE, 4.5);
    table.next(SUPERMAN_RETURNS, 4.0);
    table.next(YOU_ME_AND_DUPREE, 1.0);

    table.start();

    assert_eq!(pearson(table.get_row(0), table.get_row(1)), 0.396059017191);
    assert_eq!(pearson(table.get_row(6), table.get_row(0)), 0.99124070716192991);
    assert_eq!(pearson(table.get_row(6), table.get_row(3)), 0.89340514744156474);
    assert_eq!(pearson(table.get_row(6), table.get_row(4)), 0.92447345164190486);
}
