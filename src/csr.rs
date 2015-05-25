//! Compressed Sparse Row (and Compressed Sparse Column) implementation.

/// Value with the corresponding column.
#[derive(Debug)]
pub struct ColumnValue {
    pub column: usize,
    pub value: f64,
}

pub type Row<'a> = &'a[ColumnValue];
pub type MutableRow<'a> = &'a mut[ColumnValue];

/// Compressed Sparse Row matrix.
#[derive(Debug)]
pub struct Csr {
    values: Vec<ColumnValue>,
    pointers: Vec<usize>,
}

impl Csr {
    pub fn new() -> Self {
        Csr { values: Vec::new(), pointers: Vec::new() }
    }

    /// Starts a new row.
    ///
    /// Ensure that you have started a new row after the very last value.
    pub fn start(&mut self) {
        self.pointers.push(self.values.len());
    }

    /// Adds a new value to the current row.
    pub fn next(&mut self, column: usize, value: f64) {
        self.values.push(ColumnValue { value: value, column: column });
    }

    /// Gets value count.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Gets row count.
    pub fn row_count(&self) -> usize {
        self.pointers.len() - 1
    }

    /// Gets a slice to the row.
    pub fn get_row(&self, index: usize) -> Row {
        assert!(index < self.pointers.len());
        &self.values[self.pointers[index]..self.pointers[index + 1]]
    }

    /// Gets a mutable slice to the row.
    pub fn get_mutable_row(&mut self, index: usize) -> MutableRow {
        assert!(index < self.pointers.len());
        &mut self.values[self.pointers[index]..self.pointers[index + 1]]
    }
}

#[test]
fn test_start() {
    let mut table = Csr::new();
    table.start();
    table.next(0, 1.0);
    table.next(2, 2.0);
    table.start();

    assert_eq!(table.pointers.len(), 2);
    assert_eq!(table.pointers[0], 0);
    assert_eq!(table.pointers[1], 2);
}

#[test]
fn test_next() {
    let mut table = Csr::new();
    table.start();
    table.next(0, 1.0);
    table.next(2, 2.0);
    table.next(5, 3.0);
    table.start();

    assert_eq!(table.values.len(), 3);
}

#[test]
fn test_len() {
    let mut table = Csr::new();
    table.start();
    table.next(0, 1.0);
    table.next(2, 2.0);
    table.start();

    assert_eq!(table.len(), 2);
}

#[test]
fn test_row_count() {
    let mut table = Csr::new();
    table.start();
    table.next(0, 1.0);
    table.next(2, 2.0);
    table.start();

    assert_eq!(table.row_count(), 1);
}

#[test]
fn test_get_row() {
    let mut table = Csr::new();
    table.start();
    table.next(0, 1.0);
    table.start();
    table.next(2, 2.0);
    table.next(5, 3.0);
    table.start();
    table.next(1, 7.0);
    table.start();

    assert_eq!(table.get_row(1).len(), 2);
    assert_eq!(table.get_row(1)[0].column, 2);
}
