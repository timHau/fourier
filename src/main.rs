use fourier::fft_real;
use ndarray::array;

fn main() {
    let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let res_fft = fft_real(&signal);

    println!("{:?}", res_fft);
}