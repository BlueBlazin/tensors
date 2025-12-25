pub struct IndexGen {
    shape: Vec<usize>,
    index: Vec<usize>,
    started: bool,
    finished: bool,
}

impl IndexGen {
    pub fn new(shape: Vec<usize>) -> Self {
        let index = vec![0; shape.len()];

        Self {
            shape,
            index,
            started: false,
            finished: false,
        }
    }

    pub fn next(&mut self) -> Option<&[usize]> {
        if self.finished {
            return None;
        }

        if !self.started {
            self.started = true;
            return Some(&self.index);
        }

        let mut i = self.shape.len() - 1;

        loop {
            self.index[i] += 1;
            if self.index[i] != self.shape[i] {
                return Some(&self.index);
            }

            if i == 0 {
                self.finished = true;
                return None;
            }

            self.index[i] = 0;
            i -= 1;
        }
    }

    pub fn map_iter<F, T>(self, f: F) -> IndexGenMap<F>
    where
        F: FnMut(&[usize]) -> T,
    {
        IndexGenMap { generator: self, f }
    }
}

pub struct IndexGenMap<F> {
    generator: IndexGen,
    f: F,
}

impl<F, T> Iterator for IndexGenMap<F>
where
    F: FnMut(&[usize]) -> T,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.generator.next()?;

        let result = (self.f)(index);

        Some(result)
    }
}
