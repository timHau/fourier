mod utils;

use num_complex::Complex64;
use std::f64::consts::PI;

pub fn dft(signal: &[Complex64]) -> Vec<Complex64> {
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

pub fn dft_real(signal: &[f64]) -> Vec<Complex64> {
    let complex_signal = signal
        .iter()
        .map(|x| Complex64::new(*x, 0.0))
        .collect::<Vec<_>>();
    return dft(&complex_signal);
}

pub fn idft(spectrum: &[Complex64]) -> Vec<Complex64> {
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

pub fn fft_step(signal: &[Complex64], stride: usize) -> Vec<Complex64> {
    let n = signal.len();
    if n == 1 {
        return signal.to_vec();
    } else {
        let mut even = vec![];
        even.reserve(n / 2);
        let mut odd = vec![];
        odd.reserve(n / 2);
        for i in 0..n {
            if i % 2 == 0 {
                even.push(signal[i]);
            } else {
                odd.push(signal[i]);
            }
        }

        let even_parts = fft_step(&even, stride * 2);
        let odd_parts = fft_step(&odd, stride * 2);
        let mut result = vec![Complex64::new(0.0, 0.0); n];
        for k in 0..n / 2 {
            let angle = 2.0 * PI * (k as f64) / (n as f64);
            let c = Complex64::new(angle.cos(), -angle.sin());
            result[k] = even_parts[k] + c * odd_parts[k];
            result[k + n / 2] = even_parts[k] - c * odd_parts[k];
        }
        result
    }
}

pub fn fft(signal: &[Complex64]) -> Vec<Complex64> {
    let n = signal.len();
    let mut signal = signal.to_vec();
    if !n.is_power_of_two() {
        // extend periodically
        signal.resize(n.next_power_of_two(), Complex64::default());
        for i in n..n.next_power_of_two() {
            signal[i] = signal[i % n];
        }
    }
    fft_step(&signal, 1)
}

pub fn fft_real(signal: &[f64]) -> Vec<Complex64> {
    let complex_signal = signal
        .iter()
        .map(|x| Complex64::new(*x, 0.0))
        .collect::<Vec<_>>();
    fft(&complex_signal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn compare_vecs(result: &[Complex64], expect: &[Complex64]) {
        let eps = 0.00000001;
        result.iter().zip(expect.iter()).for_each(|(x, y)| {
            assert!((x.re.abs() - y.re.abs()).abs() <= eps);
            assert!((x.im.abs() - y.im.abs()).abs() <= eps);
        });
    }

    #[test]
    fn example_plot_fft_real() {
        let n = 256;
        let mut signal = vec![0.0; n];

        let freq_1 = 10.0;
        let freq_2 = 100.0;
        for i in 0..n {
            signal[i] += f64::sin(2.0 * PI * freq_1 * (i as f64) / (n as f64))
                + f64::sin(2.0 * PI * freq_2 * (i as f64) / (n as f64));
        }

        let spectrum = fft_real(&signal);
        assert!(utils::plot_spectrum(&spectrum, "spectrum_fft.png").is_ok());
    }

    #[test]
    fn example_plot_dft() {
        let n = 256;
        let mut signal = vec![0.0; n];

        let freq_1 = 10.0;
        let freq_2 = 100.0;
        for i in 0..n {
            signal[i] += f64::sin(2.0 * PI * freq_1 * (i as f64) / (n as f64))
                + f64::sin(2.0 * PI * freq_2 * (i as f64) / (n as f64));
        }

        assert!(utils::plot_signal(&signal, "signal.png", -2.0..2.0).is_ok());

        let spectrum = dft_real(&signal);
        assert!(utils::plot_spectrum(&spectrum, "spectrum_dft.png").is_ok());
    }

    #[test]
    fn simple_dft_test() {
        let f = [
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
        compare_vecs(&result, &expect);
    }

    #[test]
    fn dft_test() {
        // test against SciPy https://docs.scipy.org/doc/scipy/tutorial/fft.html
        let f = [
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
        compare_vecs(&result, &expect);
    }

    #[test]
    fn idft_test() {
        let f = [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(1.5, 0.0),
        ];
        let spectrum = dft(&f);
        let signal = idft(&spectrum);
        compare_vecs(&signal, &f);
    }
}
