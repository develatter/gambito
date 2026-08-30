//! Hand-rolled f32 inference for the .cyw network. No dependencies, no
//! BatchNorm (folded at export), no SIMD tricks — ~150 lines of loops that
//! mirror `forward_dequant` in python/gambito_train/export.py exactly.
//! Format: docs/encoding.md, "Model file".

use crate::encode::{encode_planes, policy_index, PLANE_COUNT, POLICY_SIZE};
use crate::eval::Evaluator;
use gambito_engine::{Move, Position};

/// The trained network ships inside the binary; gambito stays a single file.
pub const EMBEDDED_MODEL: &[u8] = include_bytes!("../model.cyw");

struct Conv {
    w: Vec<f32>, // [out][in][k][k], dequantized
    b: Vec<f32>,
    in_c: usize,
    out_c: usize,
    k: usize,
}

impl Conv {
    /// Same-padding convolution over the fixed 8x8 board. The input is
    /// copied into a zero-padded 10x10 frame per channel so the hot loop
    /// has no edge branches, and the unrolled 3x3 taps run over contiguous
    /// slices the compiler can vectorize.
    fn apply(&self, input: &[f32], relu: bool) -> Vec<f32> {
        if self.k == 1 {
            return self.apply_1x1(input, relu);
        }
        let mut padded = vec![0.0f32; self.in_c * 100];
        for i in 0..self.in_c {
            for y in 0..8 {
                padded[i * 100 + (y + 1) * 10 + 1..][..8]
                    .copy_from_slice(&input[i * 64 + y * 8..][..8]);
            }
        }
        let mut out = vec![0.0f32; self.out_c * 64];
        for o in 0..self.out_c {
            let out_ch = &mut out[o * 64..][..64];
            out_ch.fill(self.b[o]);
            for i in 0..self.in_c {
                let k = &self.w[(o * self.in_c + i) * 9..][..9];
                let p = &padded[i * 100..][..100];
                for y in 0..8 {
                    let r0 = &p[y * 10..][..10];
                    let r1 = &p[(y + 1) * 10..][..10];
                    let r2 = &p[(y + 2) * 10..][..10];
                    let dst = &mut out_ch[y * 8..][..8];
                    for x in 0..8 {
                        dst[x] += k[0] * r0[x] + k[1] * r0[x + 1] + k[2] * r0[x + 2]
                            + k[3] * r1[x] + k[4] * r1[x + 1] + k[5] * r1[x + 2]
                            + k[6] * r2[x] + k[7] * r2[x + 1] + k[8] * r2[x + 2];
                    }
                }
            }
            if relu {
                for v in out_ch.iter_mut() {
                    *v = v.max(0.0);
                }
            }
        }
        out
    }

    /// 1x1 convolution: a per-pixel linear mix of channels.
    fn apply_1x1(&self, input: &[f32], relu: bool) -> Vec<f32> {
        let mut out = vec![0.0f32; self.out_c * 64];
        for o in 0..self.out_c {
            let out_ch = &mut out[o * 64..][..64];
            out_ch.fill(self.b[o]);
            for i in 0..self.in_c {
                let wv = self.w[o * self.in_c + i];
                for (dst, src) in out_ch.iter_mut().zip(&input[i * 64..][..64]) {
                    *dst += wv * src;
                }
            }
            if relu {
                for v in out_ch.iter_mut() {
                    *v = v.max(0.0);
                }
            }
        }
        out
    }
}

struct Linear {
    w: Vec<f32>, // [out][in]
    b: Vec<f32>,
    in_d: usize,
    out_d: usize,
}

impl Linear {
    fn apply(&self, input: &[f32], relu: bool) -> Vec<f32> {
        (0..self.out_d)
            .map(|o| {
                let row = &self.w[o * self.in_d..(o + 1) * self.in_d];
                let acc: f32 = self.b[o]
                    + row.iter().zip(input).map(|(w, x)| w * x).sum::<f32>();
                if relu {
                    acc.max(0.0)
                } else {
                    acc
                }
            })
            .collect()
    }
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos + n;
        let s = self.data.get(self.pos..end).ok_or("model file truncated")?;
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        self.bytes(n * 4)?
            .chunks_exact(4)
            .map(|c| Ok(f32::from_le_bytes(c.try_into().unwrap())))
            .collect()
    }

    /// i8 weight blob followed by its scale, dequantized to f32.
    fn weights(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let raw = self.bytes(n)?;
        let scale = self.f32()?;
        Ok(raw.iter().map(|&b| b as i8 as f32 * scale).collect())
    }

    fn conv(&mut self, out_c: usize, in_c: usize, k: usize) -> Result<Conv, String> {
        let w = self.weights(out_c * in_c * k * k)?;
        let b = self.f32s(out_c)?;
        Ok(Conv { w, b, in_c, out_c, k })
    }

    fn linear(&mut self, out_d: usize, in_d: usize) -> Result<Linear, String> {
        let w = self.weights(out_d * in_d)?;
        let b = self.f32s(out_d)?;
        Ok(Linear { w, b, in_d, out_d })
    }
}

pub struct Network {
    stem: Conv,
    blocks: Vec<(Conv, Conv)>,
    policy_conv: Conv,
    policy_fc: Linear,
    value_conv: Conv,
    value_fc1: Linear,
    value_fc2: Linear,
}

impl Network {
    pub fn load(data: &[u8]) -> Result<Network, String> {
        let mut cur = Cursor { data, pos: 0 };
        if cur.bytes(4)? != b"CYW1" {
            return Err("bad magic: not a .cyw file".into());
        }
        let c = cur.u32()? as usize;
        let blocks = cur.u32()? as usize;
        let pc = cur.u32()? as usize;
        let vc = cur.u32()? as usize;
        let vh = cur.u32()? as usize;

        let net = Network {
            stem: cur.conv(c, PLANE_COUNT, 3)?,
            blocks: (0..blocks)
                .map(|_| Ok((cur.conv(c, c, 3)?, cur.conv(c, c, 3)?)))
                .collect::<Result<_, String>>()?,
            policy_conv: cur.conv(pc, c, 1)?,
            policy_fc: cur.linear(POLICY_SIZE, pc * 64)?,
            value_conv: cur.conv(vc, c, 1)?,
            value_fc1: cur.linear(vh, vc * 64)?,
            value_fc2: cur.linear(1, vh)?,
        };
        if cur.pos != data.len() {
            return Err(format!("{} trailing bytes in model file", data.len() - cur.pos));
        }
        Ok(net)
    }

    /// Raw policy logits (4168, unmasked) and tanh value for one position.
    pub fn forward(&self, planes: &[f32]) -> (Vec<f32>, f32) {
        let mut x = self.stem.apply(planes, true);
        for (c1, c2) in &self.blocks {
            let y = c2.apply(&c1.apply(&x, true), false);
            for (xi, yi) in x.iter_mut().zip(y) {
                *xi = (*xi + yi).max(0.0);
            }
        }
        let policy = self.policy_fc.apply(&self.policy_conv.apply(&x, true), false);
        let v = self.value_fc1.apply(&self.value_conv.apply(&x, true), true);
        let value = self.value_fc2.apply(&v, false)[0].tanh();
        (policy, value)
    }
}

/// The network as the MCTS sees it: the real `Evaluator` behind the seam.
pub struct NnEval {
    net: Network,
}

impl NnEval {
    /// Loads the network embedded in the binary. Panics only if the baked-in
    /// model is corrupt, which a golden test would catch long before release.
    pub fn embedded() -> NnEval {
        NnEval { net: Network::load(EMBEDDED_MODEL).expect("embedded model.cyw") }
    }

    pub fn network(&self) -> &Network {
        &self.net
    }
}

impl Evaluator for NnEval {
    fn evaluate(&self, pos: &Position, moves: &[Move]) -> (Vec<f32>, f32) {
        let (logits, value) = self.net.forward(&encode_planes(pos));
        // Masked softmax: only legal moves' logits participate, gathered by
        // policy index; max-subtraction keeps exp() finite (stable softmax).
        let gathered: Vec<f32> =
            moves.iter().map(|mv| logits[policy_index(*mv, pos.side_to_move)]).collect();
        let max = gathered.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut priors: Vec<f32> = gathered.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = priors.iter().sum();
        if sum > 0.0 {
            for p in &mut priors {
                *p /= sum;
            }
        }
        (priors, value)
    }
}
