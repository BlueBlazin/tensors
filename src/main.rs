use std::error::Error;
use tensors::{Ax, Tensor, dims, einsum, einsum2, parse};

fn main() -> Result<(), Box<dyn Error>> {
    let x: Tensor<f32> = Tensor::<f32>::randn(&[2, 4, 2]);
    let y: Tensor<f32> = Tensor::<f32>::randn(&[2, 2, 3]);

    let z = einsum!("bij,bjk->ik", &x, &y);

    println!("{:?}", z.shape());

    let x: Tensor<f32> = Tensor::<f32>::randn(&[2, 2, 2]);
    let y: Tensor<f32> = Tensor::<f32>::randn(&[3, 2, 2]);

    let z = einsum!("aii,bii", &x, &y);

    println!("{:?}", z.shape());

    let x: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let y: Tensor<f32> = Tensor::from_data(&[0.0, 100.0, 200.0, 0.0], &[1, 2, 2]);

    let z = einsum!("bii,bii->b", &x, &y);

    println!("{}", x);
    println!("{}", y);
    println!("{}", z);

    Ok(())
}
