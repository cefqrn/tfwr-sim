pub mod parser;

use parser::Parser;

fn main() {
    println!("{:?}", "pineapple".try_parse("pineapple pizza"));
    println!("{:?}", "pineapple".at_least_one().try_parse(""));
    println!(
        "{:?}",
        "pineapple".at_least_one().try_parse("pineapplepineapple")
    );
    println!(
        "{:?}",
        parser::number
            .followed_by(&parser::spaces)
            .and(&parser::number)
            .followed_by(&parser::spaces)
            .and(&parser::number)
            .try_parse("51.23     1234    .5")
    );
    println!("{:?}", parser::number.try_parse("."));
    println!("{:?}", parser::number.try_parse("5."));
    println!("{:?}", parser::identifier.try_parse("pineapple "));
    println!("{:?}", parser::identifier.try_parse("1pineapple "));
    println!(
        "{:?}",
        parser::identifier.try_parse("_ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω123 ")
    );
    println!(
        "{:?}",
        parser::identifier.try_parse("ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω123 ")
    );
    println!("{:?}", parser::expression.try_parse("None"));
    println!("{:?}", parser::expression.try_parse("Noneα"));
    println!("{:?}", parser::expression.try_parse("None "));
}
