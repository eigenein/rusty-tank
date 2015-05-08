//! Compressed Sparse Row (and Compressed Sparse Column) implementation.

/// Value with the corresponding column.
#[derive(Debug)]
pub struct ColumnValue<I, V> {
    pub column: I,
    pub value: V,
}

/// Compressed Sparse Row matrix.
#[derive(Debug)]
pub struct Csr<I, V> {
    values: Vec<ColumnValue<I, V>>,
    pointers: Vec<usize>,
}

impl<I, V> Csr<I, V> {
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
    pub fn next(&mut self, column: I, value: V) {
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
    pub fn get_row(&self, index: usize) -> &[ColumnValue<I, V>] {
        &self.values[self.pointers[index]..self.pointers[index + 1]]
    }

    /// Gets an iterator over the matrix.
    pub fn iter(&self) -> CsrIterator<I, V> {
        CsrIterator { csr: self, index: 0 }
    }
}

/// Value with the corresponding row and column.
#[derive(Debug)]
pub struct MatrixValue<I, V> {
    row: usize,
    column_value: ColumnValue<I, V>,
}

/// Iterates over CSR matrix.
pub struct CsrIterator<'a, I: 'a, V: 'a> {
    csr: &'a Csr<I, V>,
    index: usize,
}

impl<'a, I, V> Iterator for CsrIterator<'a, I, V> {
    type Item = MatrixValue<I, V>;

    fn next(&mut self) -> Option<Self::Item> {
        None // TODO: return the next item or None.
    }
}

#[test]
fn test_start() {
    let mut storage = Csr::new();
    storage.start();
    storage.next(0, 1.0);
    storage.next(2, 2.0);
    storage.start();

    assert_eq!(storage.pointers.len(), 2);
    assert_eq!(storage.pointers[0], 0);
    assert_eq!(storage.pointers[1], 2);
}

#[test]
fn test_next() {
    let mut storage = Csr::new();
    storage.start();
    storage.next(0, 1.0);
    storage.next(2, 2.0);
    storage.next(5, 3.0);
    storage.start();

    assert_eq!(storage.values.len(), 3);
}

#[test]
fn test_len() {
    let mut storage = Csr::new();
    storage.start();
    storage.next(0, 1.0);
    storage.next(2, 2.0);
    storage.start();

    assert_eq!(storage.len(), 2);
}

#[test]
fn test_row_count() {
    let mut storage = Csr::new();
    storage.start();
    storage.next(0, 1.0);
    storage.next(2, 2.0);
    storage.start();

    assert_eq!(storage.row_count(), 1);
}

#[test]
fn test_get_row() {
    let mut storage = Csr::new();
    storage.start();
    storage.next(0, 1.0);
    storage.start();
    storage.next(2, 2.0);
    storage.next(5, 3.0);
    storage.start();
    storage.next(1, 7.0);
    storage.start();

    assert_eq!(storage.get_row(1).len(), 2);
    assert_eq!(storage.get_row(1)[0].column, 2);
}
