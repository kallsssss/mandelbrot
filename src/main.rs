fn factorial(k: i32) -> i32 {
    if k != 0 {
        k * factorial(k - 1)
    } else {
        1 // base case: 0! = 1
    }
}

fn main() {
    print!("{}", factorial(10)); // 3628800
}
