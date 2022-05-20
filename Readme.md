# Fourier Transform 

This Implementation is based on the _2_-radix Cooley-Tukey FFT algorithm. So keep in mind that the input data must be a power of 2 (for now).
I also plan to implement the Cooley-Tuckey for [general Factorizations](https://numericalrecipes.wordpress.com/2009/05/29/the-cooley-tukey-fft-algorithm-for-general-factorizations/).

### Example Usage

This library uses the [ndarry](https://github.com/rust-ndarray/ndarray) crate to handle multidimensional arrays. There is distinction between real- and complex-valued data. To represent complex numbers the [num_complex](https://crates.io/crates/num_complex) crate is used. 

```rust
use fourier::{fft_real, fft};
use ndarray::array;
use num_complex::Complex64;

// if you have a real valued signal
let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let res_fft = fft_real(&signal);

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
let res_fft = fft(&signal);
```

Feel free to open an issue if you have any recommendation or find an error in my code.

If you are looking for a fast FFT implementation in Rust, I recommend the [RustFFT](https://github.com/ejmahler/RustFFT) library.