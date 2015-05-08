//! Compressed Sparse Row (and Compressed Sparse Column) implementation.

#[derive(Debug)]
pub struct IndexedValue<I, V> {
    index: I,
    value: V,
}

#[derive(Debug)]
pub struct CSR<I, V> {
    values: Vec<IndexedValue<I, V>>,
    pointers: Vec<usize>,
}

impl<I, V> CSR<I, V> {
    pub fn new() -> CSR<I, V> {
        CSR { values: Vec::new(), pointers: Vec::new() }
    }

    /// Starts a new row.
    ///
    /// Ensure that you have started a new row after the very last value.
    pub fn start(&mut self) {
        self.pointers.push(self.values.len());
    }

    /// Adds a new value to the current row.
    pub fn next(&mut self, index: I, value: V) {
        self.values.push(IndexedValue { value: value, index: index });
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
    pub fn get_row(&self, index: usize) -> &[IndexedValue<I, V>] {
        &self.values[self.pointers[index]..self.pointers[index + 1]]
    }
}

#[test]
fn test_start() {
    let mut storage = CSR::new();
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
    let mut storage = CSR::new();
    storage.start();
    storage.next(0, 1.0);
    storage.next(2, 2.0);
    storage.next(5, 3.0);
    storage.start();

    assert_eq!(storage.values.len(), 3);
}

#[test]
fn test_len() {
    let mut storage = CSR::new();
    storage.start();
    storage.next(0, 1.0);
    storage.next(2, 2.0);
    storage.start();

    assert_eq!(storage.len(), 2);
}

#[test]
fn test_row_count() {
    let mut storage = CSR::new();
    storage.start();
    storage.next(0, 1.0);
    storage.next(2, 2.0);
    storage.start();

    assert_eq!(storage.row_count(), 1);
}

#[test]
fn test_get_row() {
    let mut storage = CSR::new();
    storage.start();
    storage.next(0, 1.0);
    storage.start();
    storage.next(2, 2.0);
    storage.next(5, 3.0);
    storage.start();
    storage.next(1, 7.0);
    storage.start();

    assert_eq!(storage.get_row(1).len(), 2);
    assert_eq!(storage.get_row(1)[0].index, 2);
}
