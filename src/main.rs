fn main() {
    for i in 1..11 {
        if i == 5 {
            continue;
        }
        println!("{}", i);
    }
}
