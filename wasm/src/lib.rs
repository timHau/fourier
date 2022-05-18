mod utils;

use fourier::fft2_real;
use ndarray::Array1;
use wasm_bindgen::prelude::*;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub fn greet() {
    utils::set_panic_hook();

    let signal = ndarray![
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ];

    let res = fft2_real(&signal);
}
