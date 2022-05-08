use num_complex::Complex64;
use plotters::prelude::*;
use std::ops::Range;

pub fn plot_signal(
    signal: &[f64],
    title: &str,
    y_range: Range<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(title, (640, 420)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0f64..(signal.len() as f64), y_range)?;

    chart
        .configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .x_desc("x")
        .y_desc("y")
        .disable_mesh()
        .draw()?;

    chart.draw_series(LineSeries::new(
        signal.iter().enumerate().map(|(i, s)| (i as f64, *s)),
        &RED,
    ))?;

    chart.configure_series_labels().draw()?;

    root.present()?;
    Ok(())
}

pub fn fftshift_1d(spectrum: &[f64]) -> Vec<f64> {
    let n = spectrum.len();
    let mut spectrum = spectrum.to_vec();
    spectrum.rotate_right(n / 2);
    spectrum
}

pub fn absolute_spectrum(spectrum: &[Complex64]) -> Vec<f64> {
    spectrum.iter().map(|x| x.norm()).collect::<Vec<_>>()
}

pub fn plot_spectrum(
    spectrum: &[Complex64],
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = spectrum.len();
    // we are only in the absolute value of the spectrum
    let mut abs_spectrum = absolute_spectrum(spectrum);
    // we need to shift the spectrum to the center
    abs_spectrum = fftshift_1d(&abs_spectrum);
    // the spectrum is symmetric around 0.0 we only need the positive part
    abs_spectrum = abs_spectrum.into_iter().take(n / 2).collect::<Vec<_>>();
    abs_spectrum.reverse();

    plot_signal(&abs_spectrum, title, -10.0..300.0)
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;
    use super::*;

    #[test]
    fn abs_spec() {
        let spec = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(3.0, -4.0),
            Complex64::new(-6.0, -8.0),
            Complex64::new(15.0, 20.0),
        ];
        let abs_spec = absolute_spectrum(&spec);
        let expect = vec![0.0, 5.0, 10.0, 25.0];
        assert_eq!(abs_spec, expect);
    }

  #[test]
    fn fftshift_1() {
        let signal = [0.0, 1.0, 2.0, 3.0, 4.0, -5.0, -4.0, -3.0, -2.0, -1.0];
        let shifted = fftshift_1d(&signal);
        assert_eq!(
            shifted,
            [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        );
    }

}
