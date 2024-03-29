#[cfg(not(target_arch = "wasm32"))]
mod utils;

use ndarray::{Array1, Array2, ArrayView};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Discrete Fourier Transform (Complex valued)
pub fn dft(signal: &Array1<Complex64>) -> Vec<Complex64> {
    let n = signal.len();
    let mut result = vec![Complex64::new(0.0, 0.0); n];
    for j in 0..n {
        for k in 0..n {
            let angle = 2.0 * PI * (j * k) as f64 / (n as f64);
            let c = Complex64::new(angle.cos(), -angle.sin());
            result[j] += signal[k] * c;
        }
    }
    result
}

/// Discrete Fourier Transform (Real valued)
pub fn dft_real(signal: &Array1<f64>) -> Vec<Complex64> {
    let complex_signal = signal.mapv(|x| Complex64::new(x, 0.0));
    return dft(&complex_signal);
}

/// Inverse Discrete Fourier Transform (Complex valued)
pub fn idft(spectrum: &Array1<Complex64>) -> Vec<Complex64> {
    let n = spectrum.len();
    let mut result = vec![Complex64::new(0.0, 0.0); n];
    for j in 0..n {
        for k in 0..n {
            let angle = 2.0 * PI * (j * k) as f64 / (n as f64);
            let c = Complex64::new(angle.cos(), angle.sin());
            result[j] += spectrum[k] * c;
        }
        result[j] /= n as f64;
    }
    result
}

fn fft_step(signal: &Array1<Complex64>) -> Vec<Complex64> {
    let n = signal.len();
    if n == 1 {
        return signal.to_vec();
    } else {
        let even = Array1::from_iter(signal.iter().copied().step_by(2));
        let odd = Array1::from_iter(signal.iter().copied().skip(1).step_by(2));

        let even_parts = fft_step(&even);
        let odd_parts = fft_step(&odd);
        let mut result = vec![Complex64::default(); n];
        for k in 0..(n / 2) {
            let angle = (2.0 * PI * (k as f64)) / (n as f64);
            let c = Complex64::new(angle.cos(), -angle.sin()) * odd_parts[k];
            result[k] = even_parts[k] + c;
            result[k + n / 2] = even_parts[k] - c;
        }
        result
    }
}

/// Fast Fourier Transform (Complex valued)
pub fn fft(signal: &Array1<Complex64>) -> Vec<Complex64> {
    let n = signal.len();
    assert!(n.is_power_of_two());
    fft_step(&signal)
}

fn ifft_step(signal: &Array1<Complex64>) -> Vec<Complex64> {
    let n = signal.len();
    if n == 1 {
        return signal.to_vec();
    } else {
        let even = Array1::from_iter(signal.iter().copied().step_by(2));
        let odd = Array1::from_iter(signal.iter().copied().skip(1).step_by(2));

        let even_parts = ifft_step(&even);
        let odd_parts = ifft_step(&odd);
        let mut result = vec![Complex64::default(); n];
        for k in 0..(n / 2) {
            let angle = (2.0 * PI * (k as f64)) / (n as f64);
            let c = Complex64::new(angle.cos(), angle.sin()) * odd_parts[k];
            result[k] = even_parts[k] + c;
            result[k + n / 2] = even_parts[k] - c;
        }
        result
    }
}

/// Inverse Fast Fourier Transform (Complex valued)
pub fn ifft(spectrum: &Array1<Complex64>) -> Vec<Complex64> {
    let n = spectrum.len();
    assert!(n.is_power_of_two());
    ifft_step(&spectrum).iter().map(|x| x / n as f64).collect()
}

/// Inverse Fast Fourier Transform (Real valued)
pub fn fft_real(signal: &Array1<f64>) -> Vec<Complex64> {
    let complex_signal = signal.mapv(|x| Complex64::new(x, 0.0));
    fft(&complex_signal)
}

/// 2D Fast Fourier Transform (Complex valued)
pub fn fft2(signal: &Array2<Complex64>) -> Array2<Complex64> {
    let shape = signal.shape();
    let (nx, ny) = (shape[0], shape[1]);
    let mut fft_cols = Array2::<Complex64>::zeros((nx, 0));
    for col in signal.columns() {
        let fft_col = fft(&col.to_owned());
        fft_cols.push_column(ArrayView::from(&fft_col)).unwrap();
    }

    let mut fft_rows = Array2::<Complex64>::zeros((0, ny));
    for row in fft_cols.rows() {
        let fft_row = fft(&row.to_owned());
        fft_rows.push_row(ArrayView::from(&fft_row)).unwrap();
    }

    fft_rows
}

/// 2D Fast Fourier Transform (Real valued)
pub fn fft2_real(signal: &Array2<f64>) -> Array2<Complex64> {
    let signal = signal.mapv(|x| Complex64::new(x, 0.0));
    fft2(&signal)
}

/// Inverse 2D Fast Fourier Transform (Complex valued)
pub fn ifft2(spectrum: &Array2<Complex64>) -> Array2<Complex64> {
    let shape = spectrum.shape();
    let (nx, ny) = (shape[0], shape[1]);
    let mut ifft_cols = Array2::<Complex64>::zeros((nx, 0));
    for col in spectrum.columns() {
        let ifft_col = ifft(&col.to_owned());
        ifft_cols.push_column(ArrayView::from(&ifft_col)).unwrap();
    }

    let mut ifft_rows = Array2::<Complex64>::zeros((0, ny));
    for row in ifft_cols.rows() {
        let ifft_row = ifft(&row.to_owned());
        ifft_rows.push_row(ArrayView::from(&ifft_row)).unwrap();
    }

    ifft_rows
}

pub fn fftshift_1d(spectrum: &Array1<Complex64>) -> Vec<Complex64> {
    let n = spectrum.len();
    let mut spectrum = spectrum.to_vec();
    spectrum.rotate_right(n / 2);
    spectrum
}

pub fn fftshift_1d_real(spectrum: &Array1<f64>) -> Vec<f64> {
    let n = spectrum.len();
    let mut spectrum = spectrum.to_vec();
    spectrum.rotate_right(n / 2);
    spectrum
}

pub fn fftshift2(spectrum: &Array2<Complex64>) -> Array2<Complex64> {
    let shape = spectrum.shape();
    let (nx, ny) = (shape[0], shape[1]);

    let mut colums = Array2::<Complex64>::zeros((nx, 0));
    for col in spectrum.columns() {
        let mut v = col.to_vec();
        v.rotate_right(nx / 2);
        colums.push_column(ArrayView::from(&v)).unwrap();
    }

    let mut rows = Array2::<Complex64>::zeros((0, ny));
    for row in colums.rows() {
        let mut v = row.to_vec();
        v.rotate_right(ny / 2);
        rows.push_row(ArrayView::from(&v)).unwrap();
    }

    rows
}

pub fn fftshift2_real(spectrum: &Array2<f64>) -> Array2<f64> {
    let spectrum = spectrum.mapv(|x| Complex64::new(x, 0.0));
    let spectrum = fftshift2(&spectrum);
    spectrum.mapv(|x| x.re)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array};
    use num_complex::Complex64;

    fn compare_complex_vecs(result: &[Complex64], expect: &[Complex64]) {
        let eps = 0.00000001;
        result.iter().zip(expect.iter()).for_each(|(x, y)| {
            assert!((x.re.abs() - y.re.abs()).abs() <= eps);
            assert!((x.im.abs() - y.im.abs()).abs() <= eps);
        });
    }

    fn compare_real_vecs(result: &[f64], expect: &[f64]) {
        let eps = 0.00000001;
        result.iter().zip(expect.iter()).for_each(|(x, y)| {
            assert!((x.abs() - y.abs()).abs() <= eps);
            assert!((x.abs() - y.abs()).abs() <= eps);
        });
    }

    #[test]
    fn test_fft2() {
        let signal = array![
            [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
            [Complex64::new(5.0, 0.0), Complex64::new(6.0, 0.0)],
            [Complex64::new(7.0, 0.0), Complex64::new(8.0, 0.0)],
        ];
        let res = fft2(&signal);

        let expect = [
            [Complex64::new(36.0, 0.0), Complex64::new(-4.0, 0.0)],
            [Complex64::new(-8.0, 8.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(-8.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(-8.0, 8.0), Complex64::new(0.0, 0.0)],
        ];

        for (i, e) in expect.iter().enumerate() {
            let s = res.row(i);
            compare_complex_vecs(&s.to_vec(), e);
        }
    }

    #[test]
    fn test_ifft2() {
        let signal = array![
            [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
            [Complex64::new(5.0, 0.0), Complex64::new(6.0, 0.0)],
            [Complex64::new(7.0, 0.0), Complex64::new(8.0, 0.0)],
        ];
        let f = fft2(&signal);
        let res = ifft2(&f);

        for i in 0..4 {
            let s = res.row(i);
            println!("s = {:?}, signal = {:?}", s, signal.row(i));
            compare_complex_vecs(&s.to_vec(), &signal.row(i).to_vec());
        }
    }

    #[test]
    fn dft_fft_real_power_of_two() {
        let signal = array![1.0, 2.0, 3.0, 4.0];
        let res_fft = fft_real(&signal);
        let res_dft = dft_real(&signal);
        compare_complex_vecs(&res_fft, &res_dft);
    }

    #[test]
    fn fft_real_test() {
        let signal = array![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let res = fft_real(&signal)
            .iter()
            .map(|x| x.norm())
            .collect::<Vec<_>>();
        let expect = vec![
            4.0,
            2.613125929752753,
            0.0,
            1.0823922002923938,
            0.0,
            1.0823922002923938,
            0.0,
            2.6131259297527527,
        ];
        compare_real_vecs(&res, &expect);
    }

    #[test]
    fn dft_size() {
        let signal = array![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
        ];
        let res = dft(&signal);
        assert_eq!(res.len(), signal.len());
    }

    #[test]
    fn fft_size() {
        let signal = array![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0),
            Complex64::new(8.0, 0.0),
        ];
        let res = fft(&signal);
        assert_eq!(res.len(), signal.len());
    }

    #[test]
    fn fftshift_1_r() {
        let signal = array![0.0, 1.0, 2.0, 3.0, 4.0, -5.0, -4.0, -3.0, -2.0, -1.0];
        let shifted = fftshift_1d_real(&signal);
        assert_eq!(
            shifted,
            [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn fftshift_1_c() {
        let signal = array![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(-5.0, 0.0),
            Complex64::new(-4.0, 0.0),
            Complex64::new(-3.0, 0.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ];
        let shifted = fftshift_1d(&signal);
        let expect = array![
            Complex64::new(-5.0, 0.0),
            Complex64::new(-4.0, 0.0),
            Complex64::new(-3.0, 0.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        compare_complex_vecs(&shifted.to_vec(), &expect.to_vec());
    }

    #[test]
    fn fftshift_2_r() {
        let signal = array![[0.0, 1.0, 2.0], [3.0, 4.0, -4.0], [-1.0, -3.0, -2.0]];
        let expect = array![[-2.0, -1.0, -3.0], [2.0, 0.0, 1.0], [-4.0, 3.0, 4.0]];
        let shifted = fftshift2_real(&signal);
        assert_eq!(shifted, expect);
    }

    #[test]
    fn fftshift_2_r_2() {
        let signal = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0],
            [5.0, 6.0, 7.0, 8.0, 4.0, 3.0, 2.0, 1.0],
        ];
        let expect = array![
            [3.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            [4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0],
        ];
        let shifted = fftshift2_real(&signal);
        assert_eq!(shifted, expect);
    }

    #[test]
    fn simple_dft_test() {
        let f = array![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(-1.0, 2.0),
        ];
        let expect = [
            Complex64::new(2.0, 0.0),
            Complex64::new(-2.0, -2.0),
            Complex64::new(0.0, -2.0),
            Complex64::new(4.0, 4.0),
        ];
        let result = dft(&f);
        compare_complex_vecs(&result, &expect);
    }

    #[test]
    fn dft_test() {
        // test against SciPy https://docs.scipy.org/doc/scipy/tutorial/fft.html
        let f = array![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(1.5, 0.0),
        ];
        let expect = [
            Complex64::new(4.5, 0.0),
            Complex64::new(2.08155948, -1.65109876),
            Complex64::new(-1.83155948, 1.60822041),
            Complex64::new(-1.83155948, 1.60822041),
            Complex64::new(2.08155948, 1.65109876),
        ];
        let result = dft(&f);
        compare_complex_vecs(&result, &expect);
    }

    #[test]
    fn idft_test() {
        let f = array![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(1.5, 0.0),
        ];
        let spectrum = Array::from(dft(&f));
        let signal = idft(&spectrum);
        compare_complex_vecs(&signal, &f.to_vec());
    }

    #[test]
    fn ifft_fft() {
        let f = array![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ];
        let fft = fft(&f);
        let signal = ifft(&Array::from_vec(fft));
        compare_complex_vecs(&signal, &f.to_vec());
    }

    #[test]
    fn ifft_fft_2() {
        let f = array![
            Complex64::new(1.0, 5.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(-9.0, 0.0),
            Complex64::new(12.0, 1.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 4.0),
        ];
        let fft = fft(&f);
        let signal = ifft(&Array::from_vec(fft));
        compare_complex_vecs(&signal, &f.to_vec());
    }
}
