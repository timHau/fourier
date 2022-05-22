mod utils;

use fourier::{fft2_real, fftshift2, ifft2};
use ndarray::Array2;
use num_complex::Complex64;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct FFT2 {
    shape: (usize, usize),
    data: Array2<f64>,
}

#[wasm_bindgen]
impl FFT2 {
    pub fn init(signal: &[f64], w: usize, h: usize) -> Self {
        utils::set_panic_hook();
        Self {
            shape: (w, h),
            data: Array2::from_shape_vec((w, h), signal.iter().cloned().collect::<Vec<f64>>())
                .unwrap(),
        }
    }

    pub fn forward(&mut self) -> Vec<f64> {
        let fft_res = fft2_real(&self.data);
        let shifted = fftshift2(&fft_res);
        let mut res = vec![0.0; 2 * self.data.len()];
        for k in 0..self.data.len() {
            let (i, j) = (k / self.shape.0, k % self.shape.0);
            let c = shifted[(i, j)];
            res[2 * k] = c.re;
            res[2 * k + 1] = c.im;
        }
        res
    }

    pub fn filter(&self, spectrum: &[f64], threshold: f64, is_low_pass: bool) -> Vec<f64> {
        let (w, h) = self.shape;
        let length = spectrum.len();
        let mut res = spectrum.clone().to_vec();
        for k in 0..length / 2 {
            let (i, j) = ((k % w) as f64, (k / w) as f64);
            let (x, y) = (i - w as f64 / 2.0, j - h as f64 / 2.0);
            let dist = x.hypot(y);
            if (is_low_pass && dist > threshold) || (!is_low_pass && dist <= threshold) {
                res[2 * k] = 0.0;
                res[2 * k + 1] = 0.0;
            }
        }
        res
    }

    pub fn inverse(&self, spectrum: &[f64]) -> Vec<f64> {
        let complex_signal = spectrum
            .chunks(2)
            .map(|c| Complex64::new(c[0], c[1]))
            .collect::<Vec<Complex64>>();
        let as_array = Array2::from_shape_vec(self.shape, complex_signal).unwrap();
        let fft_res = ifft2(&as_array);
        fft_res.iter().map(|x| x.norm()).collect::<Vec<f64>>()
    }
}
