mod utils;

use fourier::{fft2_real, fftshift2};
use js_sys::Float64Array;
use ndarray::{array, Array2, ArrayView};
use wasm_bindgen::prelude::*;
use web_sys::console;

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

    pub fn forward(&self) -> Vec<f64> {
        let fft_res = fft2_real(&self.data);
        let shifted = fftshift2(&fft_res);
        let mut res = vec![0.0; self.data.len()];
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                res[i * self.shape.1 + j] = shifted[(i, j)].norm();
            }
        }
        res
    }
}
