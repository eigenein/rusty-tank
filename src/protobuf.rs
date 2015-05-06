//! Protocol Buffers format.
use std::io::Read;

/// Reads next UVarint.
pub fn read_uvarint<R: Read>(input: &mut R) -> Option<u64> {
    let mut value = 0;
    let mut shift: usize = 0;

    loop {
        let mut buffer = [0u8; 1];
        if input.read(&mut buffer).unwrap() == 0 {
            return None;
        }
        value = value | ((buffer[0] & 0x7F) as u64) << shift;
        if buffer[0] & 0x80 == 0 {
            return Some(value);
        }
        shift += 7;
    }
}

#[test]
fn test_read_uvarint() {
    use std::io::Cursor;

    assert_eq!(read_uvarint(&mut Cursor::new(vec![0x80])), None);
    assert_eq!(read_uvarint(&mut Cursor::new(vec![0x00])).unwrap(), 0);
    assert_eq!(read_uvarint(&mut Cursor::new(vec![0x03])).unwrap(), 3);
    assert_eq!(read_uvarint(&mut Cursor::new(vec![0x8E, 0x02])).unwrap(), 270);
    assert_eq!(read_uvarint(&mut Cursor::new(vec![0x9E, 0xA7, 0x05])).unwrap(), 86942);
}
