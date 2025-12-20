use std::{error::Error, fs};
use tensors::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    let data: Vec<f32> = fs::read_to_string("src/data.txt")?
        .split(',')
        .map(|s| s.trim().parse())
        .collect::<Result<Vec<_>, _>>()?;

    let mut x: Tensor<f32, 1> = Tensor::from_data(&data, &[6]);
    let y = x.reshape(&[3, 2]);
    println!("{}", x);
    x.set(&[0], 42.0);
    println!("{}", x);
    println!("{}", y);

    Ok(())
}
