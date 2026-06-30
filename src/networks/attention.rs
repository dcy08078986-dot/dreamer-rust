use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::{backend::Backend, Tensor};

/// Multi-head self-attention.
#[derive(Module, Debug)]
pub struct AttentionEncoder<B: Backend> {
    qkv: Linear<B>,
    out_proj: Linear<B>,
    ffn1: Linear<B>,
    ffn2: Linear<B>,
    out: Linear<B>,
    d_model: usize,
    num_heads: usize,
    scale: f32,
}

impl<B: Backend> AttentionEncoder<B> {
    pub fn init(
        device: &B::Device,
        d_model: usize,
        num_heads: usize,
        embed_dim: usize,
    ) -> Self {
        let d_head = d_model / num_heads;
        Self {
            qkv: LinearConfig::new(d_model, 3 * d_model).init(device),
            out_proj: LinearConfig::new(d_model, d_model).init(device),
            ffn1: LinearConfig::new(d_model, d_model * 2).init(device),
            ffn2: LinearConfig::new(d_model * 2, d_model).init(device),
            out: LinearConfig::new(d_model, embed_dim).init(device),
            d_model,
            num_heads,
            scale: (d_head as f32).sqrt(),
        }
    }

    /// Encode a sequence of tokens: [B, N, d_model] → [B, embed_dim]
    /// Uses mean-pooling over tokens (simpler and more stable than CLS).
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [b, n, d] = x.dims();

        // —— Self-attention ——
        let flat = x.clone().reshape([b * n, d]);
        let qkv = self.qkv.forward(flat);
        let chunk = self.d_model;

        let q = qkv.clone().narrow(1, 0, chunk);
        let k = qkv.clone().narrow(1, chunk, chunk);
        let v = qkv.narrow(1, 2 * chunk, chunk);

        // Reshape: [B*N, d] → [B, num_heads, N, d_head]
        let d_head = self.d_model / self.num_heads;
        let q = q.reshape([b * self.num_heads, n, d_head]);
        let k = k.reshape([b * self.num_heads, n, d_head]);
        let v = v.reshape([b * self.num_heads, n, d_head]);

        let attn = softmax(q.matmul(k.transpose()).div_scalar(self.scale), 2);
        let attn_out = attn.matmul(v).reshape([b, n, self.d_model]);

        let x = x.add(self.out_proj.forward(attn_out.reshape([b * n, self.d_model])).reshape([b, n, self.d_model]));

        // —— FFN ——
        let ff = self.ffn2.forward(relu(self.ffn1.forward(x.clone().reshape([b * n, self.d_model]))));
        let x = x.add(ff.reshape([b, n, self.d_model]));

        // —— Mean pool → embed ——
        let pooled = x.sum_dim(1).squeeze(1).div_scalar(n as f32); // [B, d_model]
        self.out.forward(pooled)
    }
}
