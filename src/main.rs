use num_complex::Complex64;
use std::f64::consts::PI;

fn dft(signal: &[Complex64]) -> Vec<Complex64> {
    let n = signal.len();
    let mut result = vec![Complex64::new(0.0, 0.0); n];
    for j in 0..n {
        for k in 0..n {
            let angle = 2.0 * PI * (j as f64 * k as f64) / (n as f64);
            let c = Complex64::new(angle.cos(), -angle.sin());
            result[j] += signal[k] * c;
        }
    }
    result
}

fn idft(spectrum: &[Complex64]) -> Vec<Complex64> {
    let n = spectrum.len();
    let mut result = vec![Complex64::new(0.0, 0.0); n];
    for j in 0..n {
        for k in 0..n {
            let angle = 2.0 * PI * (j as f64 * k as f64) / (n as f64);
            let c = Complex64::new(angle.cos(), angle.sin());
            result[j] += spectrum[k] * c;
        }
        result[j] /= n as f64;
    }
    result
}

fn main() {
    // let mut f = [1.0, 2.0, 1.0, -1.0, 1.5]
    //     .iter()
    //     .map(|&x| Complex32::new(x, 0.0))
    //     .collect::<Vec<_>>();

    let f = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(-1.0, 2.0),
    ];
    println!(
        "{:?}",
        dft(&f).iter().map(|x| x.to_string()).collect::<Vec<_>>()
    );
    println!(
        "{:?}",
        idft(&dft(&f))
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
    );
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
}
