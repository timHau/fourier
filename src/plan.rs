use std::ops::Deref;

use ndarray::Array1;

#[derive(Debug)]
pub struct Planner<T> {
    factors: (usize, usize),
    signal: Array1<T>,
}

impl<T: Clone> Planner<T> {
    pub fn new(data: &Array1<T>) -> Self {
        let prime_factors = Self::factor(data.len());

        let factors = prime_factors.split_off(prime_factors.len() / 2);

        let signal = data.to_owned();
        Self { factors, signal }
    }

    fn factor(n: usize) -> Vec<usize> {
        let mut res = vec![];
        let mut n = n;
        let mut i = 2;
        while i * i <= n {
            if n % i == 0 {
                n /= i;
                res.push(i);
            } else {
                i += 1;
            }
        }
        if n > 1 {
            res.push(n);
        }
        res
    }

    fn exec(&self) -> Array1<T> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tmp() {
        Planner::<u8>::factor(18);
        println!("{:?}", Planner::<u8>::factor(18));
    }

    #[test]
    fn prime_factors() {
        let prime_factors = Planner::<u8>::factor(12);
        assert_eq!(prime_factors, vec![2, 2, 3]);
    }

    #[test]
    fn prime_factors_2() {
        let prime_factors = Planner::<u64>::factor(1234512323545);
        assert_eq!(prime_factors, vec![5, 131, 1889, 997751]);
    }

    #[test]
    fn prime_factors_3() {
        let prime_factors = Planner::<u32>::factor(560);
        assert_eq!(prime_factors, vec![2, 2, 2, 2, 5, 7]);
    }
}
