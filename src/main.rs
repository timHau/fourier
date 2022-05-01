use num_complex::Complex32;
use std::f32::consts::PI;

fn dft(signal: &[Complex32]) -> Vec<Complex32> {
    let n = signal.len();
    let mut result = vec![Complex32::new(0.0, 0.0); n];
    for j in 0..n {
        for k in 0..n {
            let angle = 2.0 * PI * (j as f32 * k as f32) / (n as f32);
            let c = Complex32::new(angle.cos(), -angle.sin());
            result[j] += signal[k] * c;
        }
    }
    result
}

fn idft(spectrum: &[Complex32]) -> Vec<Complex32> {
    let n = spectrum.len();
    let mut result = vec![Complex32::new(0.0, 0.0); n];
    for j in 0..n {
        for k in 0..n {
            let angle = 2.0 * PI * (j as f32 * k as f32) / (n as f32);
            let c = Complex32::new(angle.cos(), angle.sin());
            result[j] += spectrum[k] * c;
        }
        result[j] /= n as f32;
    }
    result
}


fn main() {
    // let mut f = [1.0, 2.0, 1.0, -1.0, 1.5]
    //     .iter()
    //     .map(|&x| Complex32::new(x, 0.0))
    //     .collect::<Vec<_>>();

    let mut f = [
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, -1.0),
        Complex32::new(0.0, -1.0),
        Complex32::new(-1.0, 2.0),
    ];
    println!("{:?}", dft(&f).iter().map(|x| x.to_string()).collect::<Vec<_>>());
    println!("{:?}", idft(&dft(&f)).iter().map(|x| x.to_string()).collect::<Vec<_>>());
}
