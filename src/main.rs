use fourier::{fft, fft_real};
use num_complex::Complex64;
use ndarray::array;

fn main() {
    // if you have a real valued signal
    let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let _res_fft = fft_real(&signal);

    // if you have a complex valued signal
    let signal = array![
        Complex64::new(1.0, 8.0),
        Complex64::new(2.0, -7.0),
        Complex64::new(3.0, 6.0),
        Complex64::new(4.0, -5.0),
        Complex64::new(5.0, 4.0),
        Complex64::new(6.0, -3.0),
        Complex64::new(7.0, 2.0),
        Complex64::new(8.0, -1.0)
    ];
    let _res_fft = fft(&signal);
}
