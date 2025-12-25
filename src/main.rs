use std::{error::Error, fs};
use tensors::{Ax, Tensor, dims, einsum, einsum2, parse};

fn main() -> Result<(), Box<dyn Error>> {
    // let data: Vec<f32> = fs::read_to_string("src/data.txt")?
    //     .split(',')
    //     .map(|s| s.trim().parse())
    //     .collect::<Result<Vec<_>, _>>()?;

    // let x: Tensor<f32> = Tensor::from_data(&data, &[3, 3, 2]);

    // let y: Tensor<f32> = Tensor::from_data(&vec![100.0, 200.0], &[2]);

    // let z = x.div(&y);

    // println!(
    //     "x.shape = {:?}, y.shape = {:?}, z.shape = {:?}",
    //     x.shape(),
    //     y.shape(),
    //     z.shape()
    // );

    // println!("{}", z);

    let x: Tensor<f32> = Tensor::<f32>::randn(&[2, 4, 2]);
    let y: Tensor<f32> = Tensor::<f32>::randn(&[2, 2, 3]);

    let z = einsum!("bij,bjk->ik", &x, &y);

    println!("{:?}", z.shape());

    // let x: Tensor<f32> = Tensor::<f32>::randn(&[2, 2, 2]);
    // let y: Tensor<f32> = Tensor::<f32>::randn(&[3, 2, 2]);

    // let z = einsum!("aii,bii", &x, &y);

    // println!("{:?}", z.shape());

    // let x: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    // let y: Tensor<f32> = Tensor::from_data(&[0.0, 100.0, 200.0, 0.0], &[1, 2, 2]);

    // let z = einsum!("bii,bii->b", &x, &y);

    // println!("{}", x);
    // println!("{}", y);
    // println!("{}", z);

    Ok(())
}
