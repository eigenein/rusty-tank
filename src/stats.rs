//! Statistics file reader.
use std::io::Read;

use protobuf;

#[derive(Debug)]
pub struct Tank {
    pub id: u16,
    pub battles: u32,
    pub wins: u32,
}

#[derive(Debug)]
pub struct Account {
    pub id: u32,
    pub tanks: Vec<Tank>,
}

/// Reads next account statistics.
pub fn read_account<R: Read>(input: &mut R) -> Option<Account> {
    if !skip_account_header(input) {
        return None;
    }
    let account_id = protobuf::read_uvarint(input).unwrap();
    let tank_count = protobuf::read_uvarint(input).unwrap();
    let mut tanks = Vec::new();
    for _ in 0..tank_count {
        let tank_id = protobuf::read_uvarint(input).unwrap();
        let battles = protobuf::read_uvarint(input).unwrap();
        let wins = protobuf::read_uvarint(input).unwrap();
        tanks.push(Tank { id: tank_id as u16, battles: battles as u32, wins: wins as u32 });
    }
    Some(Account { id: account_id as u32, tanks: tanks })
}

/// Skips account header.
fn skip_account_header<R: Read>(input: &mut R) -> bool {
    let mut buffer = [0u8; 1];
    input.read(&mut buffer).unwrap() == 1 && input.read(&mut buffer).unwrap() == 1
}

#[test]
fn test_read_account() {
    use std::io::Cursor;

    let account = read_account(&mut Cursor::new(vec![0x3e, 0x3e, 0x03, 0x01, 0x8E, 0x02, 0x9E, 0xA7, 0x05, 0x9D, 0xA7, 0x05])).unwrap();
    assert_eq!(account.id, 3);
    assert_eq!(account.tanks.len(), 1);
    assert_eq!(account.tanks[0].id, 270);
    assert_eq!(account.tanks[0].battles, 86942);
    assert_eq!(account.tanks[0].wins, 86941);
}
