use crate::{fft2_real, fftshift2, fftshift_1d_real};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use plotters::prelude::*;
use std::{error::Error, ops::Range};

#[allow(unused)]
pub fn plot_signal(signal: &[f64], title: &str, y_range: Range<f64>) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(title, (640, 420)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0f64..(signal.len() as f64), y_range)?;

    chart
        .configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .x_desc("time")
        .y_desc("amplitude")
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

#[allow(unused)]
pub fn absolute_spectrum(spectrum: &[Complex64]) -> Vec<f64> {
    spectrum.iter().map(|x| x.norm()).collect::<Vec<_>>()
}

#[allow(unused)]
pub fn plot_spectrum(
    spectrum: &[Complex64],
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = spectrum.len();
    // we are only in the absolute value of the spectrum
    let mut abs_spectrum = absolute_spectrum(spectrum);
    // we need to shift the spectrum to the center
    abs_spectrum = fftshift_1d_real(&Array1::from_vec(abs_spectrum));
    // the spectrum is symmetric around 0.0 we only need the positive part
    abs_spectrum = abs_spectrum.into_iter().take(n / 2).collect::<Vec<_>>();
    abs_spectrum.reverse();

    let root = BitMapBackend::new(title, (640, 420)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0f64..(abs_spectrum.len() as f64), -10.0..300.0)?;

    chart
        .configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .x_desc("frequency")
        .y_desc("amplitude")
        .disable_mesh()
        .draw()?;

    chart.draw_series(LineSeries::new(
        abs_spectrum.iter().enumerate().map(|(i, s)| (i as f64, *s)),
        &RED,
    ))?;

    chart.configure_series_labels().draw()?;

    root.present()?;
    Ok(())
}

#[allow(unused)]
pub fn plot_fft2(
    spectrum: &[f64],
    dim: (u32, u32),
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(title, dim).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(25)
        .y_label_area_size(25)
        .build_cartesian_2d(
            (-(dim.0 as f64) / 2.0)..(dim.0 as f64) / 2.0,
            -(dim.1 as f64) / 2.0..(dim.1 as f64) / 2.0,
        )?;

    chart.configure_mesh().disable_mesh().draw()?;

    let plotting_area = chart.plotting_area();

    let data = Array2::from_shape_vec((dim.0 as usize, dim.1 as usize), spectrum.to_vec()).unwrap();
    let f = fftshift2(&fft2_real(&data));
    let f_abs = f.map(|x| x.norm());

    for (pos, x) in f_abs.indexed_iter() {
        let val = (1.5 * x.log2() * x.log10()) as u8;
        plotting_area.draw_pixel(
            (
                pos.0 as f64 - dim.0 as f64 / 2.0,
                pos.1 as f64 - dim.1 as f64 / 2.0,
            ),
            &RGBColor(val, val, val),
        )?;
    }

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dft_real, fft_real};
    use image::io::Reader as ImageReader;
    use num_complex::Complex64;
    use std::f64::consts::PI;

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

        let spectrum = fft_real(&Array1::from_vec(signal));
        assert!(plot_spectrum(&spectrum, "./img/spectrum_fft.png").is_ok());
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

        assert!(plot_signal(&signal, "./img/signal.png", -2.0..2.0).is_ok());

        let spectrum = dft_real(&Array1::from_vec(signal));
        assert!(plot_spectrum(&spectrum, "./img/spectrum_dft.png").is_ok());
    }

    #[test]
    fn example_plot_fft2() {
        let mandrill = ImageReader::open("./img/mandrill.png")
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();
        let data = mandrill
            .iter()
            .step_by(3)
            .map(|x| *x as f64)
            .collect::<Vec<_>>();
        plot_fft2(&data, mandrill.dimensions(), "./img/fft2.png").expect("plotting failed");
    }

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
}
