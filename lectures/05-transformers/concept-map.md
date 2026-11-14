# Concept map: *Stanford CS224N (Winter 2026)*, Chapter 5 (Lecture 5: Transformers)


Lead: claude (sole writer). Reviewer: codex. An entry appears here only after the Reviewer has judged it.
The map is meant to be complete enough to rebuild the lesson without the source. Citations use section numbers
plus short quotes.

## Teaching arc


1. Slides 1–21 recap the recurrent LM, vanishing/exploding gradients, clipping, LSTM memory and recurrent seq2seq. The tickets dependency, translation examples, end-to-end conditional LM objective and fixed-vector bottleneck repeat ch04:C17, ch04:C18, ch04:C19, ch04:C20, ch04:C21, ch04:C24, ch04:C25, ch04:C26, ch04:C28. Slide 15 repeats the first-seq2seq wording; the priority caveat in ch04:C24 still applies.
2. Notes 10 §1 distinguishes static from contextual vectors (C01) and explains recurrence’s serial depth and interaction paths (C02), motivating attention as a replacement.

3. Pooling supplies a query-independent baseline (C03); soft lookup (C04) makes source access depend on the current decoder state. Recurrent cross-attention (C05) bypasses the fixed handoff and yields soft alignments (slides 22–38).

4. Self-attention supplies query/key/value projections from one sequence (C06); sequence matrices compute all positions in parallel (C07).

5. Order is not built into raw self-attention, so position representations supply it (C08), with learned, sinusoidal and relative variants.

6. Shared position-wise nonlinear networks process attention outputs (C09), complementing attention’s token mixing.

7. Causal masking (C10) prevents looking at targets while retaining parallel training computations. The minimal architecture now combines C06, C08, C09 and C10 (notes §2.5); the full Transformer adds the next components.

8. Multi-head attention combines separately projected contextual lookups (C11); scaling by the square root of head width helps control attention-logit magnitudes (C12).

9. Residual streams preserve inputs while sublayers add changes (C13). Layer normalization stabilizes per-token feature scales (C14); the sources distinguish pre-norm and post-norm ordering.

10. The components form causal decoder-only models (C15), bidirectional encoders (C16), or source-conditioned encoder–decoders (C17), distinguished by their allowed context and cross-attention.

11. An LM head maps final states to vocabulary probabilities (C18). Self-supervised pretraining uses shifted true targets in parallel (C19); perplexity evaluates next-token prediction with a tokenizer-dependent unit (C20).

12. Generation needs a token-selection policy (C21); temperature reshapes probabilities (C22), while top-k/top-p truncate and renormalize candidate sets (C23).

13. MT and document-generation evidence supports the architecture, while all-pairs costs motivate long-context research (C24). KV reuse saves repeated inference work (C25); source-era configurations illustrate varying widths, layer counts and context lengths.

14. The assigned image/music papers transfer the architecture beyond text: local attention makes image neighborhoods tractable (C26); relative-position computation supports musical structure without the previous gathered-tensor cost (C27).

15. J&M’s historical synthesis links speech-recognition n-grams and the language/acoustic-model distinction to maxent predictors (C28), LSA/SVD embeddings, joint neural embedding/prediction, RNN LMs and word2vec, then whole-LM transfer and GPT-2 zero-shot use. Static distributional embeddings repeat ch02:C07 and ch02:C24; foundation models repeat ch01:C44. Weighted source lookup and memory-network query/key/value vocabulary are separate ancestors of the transformer. Bibliographic lists identify sources without expanding this map to every cited paper’s independent lessons.

## Assumed background

What the chapter uses without teaching it is recorded as **studs** (B01…, below).

**Chapter scope** names what was read, including footnotes or notes wherever the converter placed them.

**Chapter scope:** the Winter 2026 Lecture 5 deck, PDF pages 1–70, from the title through the final quadratic-attention question; Note 10 (Winter 2023 draft), §1–3.7, including numbered marginal notes 1–4 and References on PDF pp. 14–18; seven assigned readings including the August 2026 J&M Chapter 7, from its introduction through §7.9, Historical Notes, Exercises 7.1–7.5 and bibliography. J&M footnotes are on their pages (including pp. 3, 12, 18 and 20). The prepared source sheet lists the cited works consulted where the lecture points. Videos are links only.

**Citation convention:** “slide” means PDF page. Printed numbering is one ahead from PDF page 39 onward. “notes” means Note 10; “J&M” means the August 19, 2026 Chapter 7. The lecture's materials are in `materials/`; the course index lists each with its source and offering. No research paper’s bibliography expands the lecture’s scope.

## Figures

Concepts cite figures by number. Images are linked, never committed.

| Fig | Shows | Teaches | Image |
|---|---|---|---|
| 2.1 | Notes Figure 1: serial operations before each recurrent state | C02 | [notes p3](materials/figures/notes-p03.png) |
| 2.2 | Notes Figure 2: operation distance from tea to other positions | C02 | [notes p4](materials/figures/notes-p04.png) |
| 1.23 | Mean/max pooling for sentence sentiment | C03 | [slide 23](materials/slides/cs224n-2026-lecture05-transformers/p23.png) |
| 1.24 | Exact table lookup versus attention’s soft weighted lookup | C04 | [slide 24](materials/slides/cs224n-2026-lecture05-transformers/p24.png) |
| 1.25 | Decoder-to-encoder dot scores | C05 | [slide 25](materials/slides/cs224n-2026-lecture05-transformers/p25.png) |
| 1.29 | Softmax distribution for the first target token | C05 | [slide 29](materials/slides/cs224n-2026-lecture05-transformers/p29.png) |
| 1.30 | Weighted source-state sum | C04, C05 | [slide 30](materials/slides/cs224n-2026-lecture05-transformers/p30.png) |
| 1.31 | Concatenate context with decoder state before prediction | C05 | [slide 31](materials/slides/cs224n-2026-lecture05-transformers/p31.png) |
| 1.32 | Input-feeding and the next target step | C05 | [slide 32](materials/slides/cs224n-2026-lecture05-transformers/p32.png) |
| 1.37 | Learned source–target soft alignment | C05 | [slide 37](materials/slides/cs224n-2026-lecture05-transformers/p37.png) |
| 1.41 | Hypothetical attention weights for learned | C06 | [1.41](materials/slides/cs224n-2026-lecture05-transformers/p41.png) |
| 1.42 | Scalar key-query-value equations | C06 | [1.42](materials/slides/cs224n-2026-lecture05-transformers/p42.png) |
| 1.56 | Sequence-matrix attention | C07 | [1.56](materials/slides/cs224n-2026-lecture05-transformers/p56.png) |
| 3.3 | J&M Figure 7.3: causal pronoun representation before disambiguation | C01, C06 | [3.3](materials/figures/jm-p04.png) |
| 1.44 | Adding position vectors to input embeddings | C08 | [slide 44](materials/slides/cs224n-2026-lecture05-transformers/p44.png) |
| 1.45 | Sinusoidal equations and frequency/position heatmap | C08 | [slide 45](materials/slides/cs224n-2026-lecture05-transformers/p45.png) |
| 1.46 | Learned absolute positions and named relative alternatives | C08 | [slide 46](materials/slides/cs224n-2026-lecture05-transformers/p46.png) |
| 1.48 | Shared FF networks after attention at each token | C09 | [1.48](materials/slides/cs224n-2026-lecture05-transformers/p48.png) |
| 3.10 | J&M’s gated FFN equation and parameter discussion | C09 | [3.10](materials/figures/jm-p10.png) |
| 1.50 | Triangular causal score mask including the diagonal | C10 | [1.50](materials/slides/cs224n-2026-lecture05-transformers/p50.png) |
| 1.52 | Minimal architecture with position, attention, FFN and mask | C10 | [1.52](materials/slides/cs224n-2026-lecture05-transformers/p52.png) |
| 2.3 | Notes Figure 3: permitted prefix at each Zuko token | C10 | [2.3](materials/figures/notes-p09.png) |
| 1.55 | Hypothetical distinct entity/syntax head preferences | C11 | [slide 55](materials/slides/cs224n-2026-lecture05-transformers/p55.png) |
| 1.57 | Per-head projections and output concatenation | C11 | [slide 57](materials/slides/cs224n-2026-lecture05-transformers/p57.png) |
| 1.58 | Head-axis batching and reduced per-head widths | C11 | [slide 58](materials/slides/cs224n-2026-lecture05-transformers/p58.png) |
| 1.59 | Scale dot-product logits by square root of head width | C12 | [slide 59](materials/slides/cs224n-2026-lecture05-transformers/p59.png) |
| 1.61 | Identity shortcut plus learned update, and illustrative ResNet landscapes | C13 | [1.61](materials/slides/cs224n-2026-lecture05-transformers/p61.png) |
| 1.62 | Per-token mean/variance, gain and bias | C14 | [1.62](materials/slides/cs224n-2026-lecture05-transformers/p62.png) |
| 3.7 | J&M Figure 7.7: pre-norm residual-stream block | C13 | [3.7](materials/figures/jm-p09.png) |
| 3.8 | J&M Figure 7.8: information moves between token streams | C13 | [3.8](materials/figures/jm-p12.png) |
| 1.63 | Causal decoder-only block stack | C15 | [slide 63](materials/slides/cs224n-2026-lecture05-transformers/p63.png) |
| 1.64 | Remove future masking for bidirectional encoder context | C16 | [slide 64](materials/slides/cs224n-2026-lecture05-transformers/p64.png) |
| 1.65 | Encoder–decoder architecture with extra cross-attention | C17 | [slide 65](materials/slides/cs224n-2026-lecture05-transformers/p65.png) |
| 1.66 | Decoder queries and final-encoder keys/values | C17 | [slide 66](materials/slides/cs224n-2026-lecture05-transformers/p66.png) |
| 3.15 | J&M Figure 7.15: LM output head and tied unembedding | C18 | [3.15](materials/figures/jm-p18.png) |
| 3.16 | J&M Figure 7.16: complete decoder-only LM for one token | C15, C18 | [3.16](materials/figures/jm-p19.png) |
| 3.22 | J&M Figure 7.22: shifted training targets and token losses | C19 | [3.22](materials/figures/jm-p26.png) |
| 3.27 | J&M likelihood, perplexity and cross-entropy equations | C20 | [3.27](materials/figures/jm-p27.png) |
| 3.18 | J&M Figure 7.18: greedy next-token choice | C21 | [3.18](materials/figures/jm-p21.png) |
| 3.19 | J&M Figure 7.19: multinomial sampling | C21 | [3.19](materials/figures/jm-p22.png) |
| 3.21 | J&M Figure 7.21: temperature distributions, with top-k/top-p discussion | C22, C23 | [3.21](materials/figures/jm-p24.png) |
| 1.67 | Prior MT systems and ensembles on WMT14; transformer rows absent | C17 | [1.67](materials/slides/cs224n-2026-lecture05-transformers/p67.png) |
| 1.68 | WikiSum perplexity/ROUGE-L across configurations | C15 | [1.68](materials/slides/cs224n-2026-lecture05-transformers/p68.png) |
| 1.69 | Quadratic attention and richer positions as open questions | C24 | [1.69](materials/slides/cs224n-2026-lecture05-transformers/p69.png) |
| 1.70 | Scaling observations and long-context question | C24 | [1.70](materials/slides/cs224n-2026-lecture05-transformers/p70.png) |
| 3.23 | J&M Figure 7.23: new query over cached and new K/V | C25 | [3.23](materials/figures/jm-p28.png) |
| 3.24 | J&M Figure 7.24: source-era model configurations | C15 | [3.24](materials/figures/jm-p29.png) |
| 4.1 | Vaswani Table 1: complexity, sequential operations and path lengths | C24 | [4.1](materials/figures/vaswani-p06.png) |
| 4.2 | Vaswani Table 2: transformer results, costs and inconsistent body number | C17 | [4.2](materials/figures/vaswani-p08.png) |
| 5.1 | Image Transformer Figure 1 and completion/super-resolution examples | C26 | [5.1](materials/figures/image-p03.png) |
| 5.2 | Image Transformer Figure 2: 1D/2D local blocks and masking | C26 | [5.2](materials/figures/image-p05.png) |
| 5.5 | Image Transformer Table 5: human evaluation and protocol | C26 | [5.5](materials/figures/image-p07.png) |
| 5.6 | Image Transformer Table 6: generated details and inconsistent temperature caption | C26 | [5.6](materials/figures/image-p08.png) |
| 6.1 | Music Transformer Figure 1: relative-logit skew | C27 | [6.1](materials/figures/music-p04.png) |
| 6.2 | Music Transformer Table 1 and Figure 2: memory accounting and local skew | C27 | [6.2](materials/figures/music-p05.png) |
| 6.7 | Music Transformer Figure 7: event serialization; listening-study protocol | C27 | [6.7](materials/figures/music-p12.png) |
| 6.8 | Music Transformer Figures 8–9: recurring notes and left-hand attention; listening comparisons | C27 | [6.8](materials/figures/music-p13.png) |

## Concepts


### C01 Contextual token representations
- **Meaning:** A token vector depends on its surrounding sequence or allowed prefix. In contrast, a non-contextual embedding depends only on the word’s identity (notes §1.1).
- **Symbols:** E ∈ R^(d×|V|); x_i = E w_i ∈ R^d. A contextual vector is h_i = f_i(x_1:n), or f_i(x_1:i) when only causal context is allowed. The notes also stack embeddings as rows, X ∈ R^(n×d).
- **Key points:** Strong contextual representations are the objective: a vector is a “function of the entire sequence” (notes §1.1). Context determines which information should influence a token; embeddings provide the initial input.
- **Example:** The Zuko sentence in notes Figures 1–2 has one representation at each position. Later attention supplies another way to combine context.
- **Extends:** ch02:C07, from a static dense vector for a word to a context-sensitive vector for a token.

- *Revised at U3 (#4936):* J&M §7.1 adds chicken/road pronoun ambiguity, keys/are agreement, and pond/bank sense examples. A causal prefix cannot use its unseen continuation.

### C02 Recurrence’s serial dependencies and linear interaction distance
- **Meaning:** Recurrence makes each hidden state wait for its predecessor, while the path carrying information between distant tokens grows with their separation. These are related computational and modeling limitations (notes §1.2).
- **Symbols:** h_t = σ(W h_(t−1) + U x_t). Computing h_2 requires h_1; substituting the recurrence makes that dependency explicit.
- **Key points:** GPU matrix products can parallelize many operations, but the state dependency chain limits parallel work across token positions. The notes call the other limitation “Linear interaction distance” (§1.2): distant lexical information passes through a growing number of matrix products and nonlinearities. Difficult learning and recall motivate more direct connections; replacing recurrence does not make autoregressive generation parallel.
- **Example:** The agreement between chef and is crosses a long intervening phrase. Notes Figures 1–2 use Zuko made his uncle tea to show serial-operation counts and operation distance.
- **Extends:** ch04:C12, adding explicit accounts of serial depth and token-interaction paths.
- **Figures:** 2.1, 2.2.

### C03 Pooling a sequence’s hidden states
- **Meaning:** Collapse the token states into a sentence vector using their elementwise mean or maximum instead of only the final state (slide 23). This baseline does not depend on a query.
- **Symbols:** Mean h̄ = (1/n) Σ_j h_j; maximum h̄_r = max_j h_(j,r).
- **Key points:** Averaging is a “very basic way” to pass information from the encoder (slide 23); elementwise maximum is another option.
- **Example:** overall I enjoyed the movie a lot is pooled for a positive sentiment prediction.
- **Figures:** 1.23.

### C04 Attention as query-dependent soft lookup
- **Meaning:** A query assigns normalized weights to keys and returns a weighted sum of their corresponding values. Unlike exact lookup, many values can contribute (slides 24, 38).
- **Symbols:** e_j = score(q,k_j); α_j = exp(e_j)/Σ_l exp(e_l); c = Σ_j α_j v_j.
- **Key points:** “Attention is just a weighted average” (slide 24). Learned weights provide flexible routing; the technique applies beyond translation and beyond seq2seq (slide 38). Keys determine relevance and values carry content, even when both come from the same hidden states.
- **Example:** A decoder query looks up source states relevant to the next target token.
- **Extends:** ch01:C40, adding the actual soft-lookup computation.
- **Figures:** 1.24, 1.30.

- *Revised at U7 (#4946):* Additive scoring computes query–key compatibility “using a feed-forward network with a single hidden layer” (Vaswani §3.2.1). Dot-product scoring is especially efficient in optimized matrix multiplication. These are alternatives inside the general soft-lookup operation.

### C05 Recurrent seq2seq with cross-attention and soft alignment
- **Meaning:** Each recurrent decoder state directly consults the entire encoder-state sequence, replacing reliance on a single fixed-vector handoff (slides 22, 25–38).
- **Symbols:** Decoder state s_t, encoder states h_j; e_(tj) = s_tᵀh_j; α_t = softmax(e_t); c_t = Σ_j α_(tj)h_j. The concatenation [s_t;c_t] is used to compute the next-token distribution (slide 31).
- **Key points:** The decoder can bypass the fixed-vector bottleneck and use shorter gradient paths; the lecture reports improved NMT and some interpretability through the soft source–target alignment (slide 37). This does not guarantee a complete explanation or perfect translation. Optional input-feeding supplies the preceding context vector to the next recurrent step (slide 32). Source–target scoring has O(nm) interactions, quadratic when both lengths grow together.
- **Example:** il a m’ entarté → he hit me with a pie. The attention distribution changes for each target word. Slide 29 labels the first source hidden state “he”; its input is actually il.
- **Extends:** ch04:C25, replacing the fixed-vector handoff by a query-dependent source lookup while retaining recurrent encoder and decoder.
- **Figures:** 1.25, 1.29, 1.30, 1.31, 1.32, 1.37. The Google blog’s Flow of Information gives related context-dependent routing examples.

### C06 Key-query-value self-attention
- **Meaning:** Each position builds a contextual vector by attending to the same sequence that supplies its own query. Separate learned projections provide the asking, matching and returning views (slide 42; notes §2.1).
- **Symbols:** With column vectors, q_i = W_Q x_i, k_j = W_K x_j, v_j = W_V x_j; e_ij = q_iᵀk_j; α_ij = exp(e_ij)/Σ_l exp(e_il); o_i = Σ_j α_ij v_j. These equations are unscaled at this stage of the lecture.
- **Key points:** The projections learn “different views” (notes §2.1) for query, key and value roles. Queries, keys and values come from one sequence, unlike recurrent cross-attention (C05). Self-attention does not inherently require causal masking.
- **Example:** Zuko made his uncle tea (slide 42). In slide 41’s hypothetical example, the query for learned emphasizes CS and learned in I went to Stanford CS 224n and learned. J&M Figure 7.3 shows a causal it representation drawing on both chicken and road before its disambiguating continuation appears.
- **Figures:** 1.41, 1.42, 3.3.

- *Revised at U7 (#4944):* C12 supplies the scaled production form; the earlier unscaled equations are pedagogical.

### C07 Sequence-matrix form and parallel computation of attention
- **Meaning:** Pack positions into rows and use matrix products to compute all projected vectors, pairwise scores and weighted outputs. Every output reads the preceding layer’s vectors, not another output in its own layer (slide 56; notes §3.1).
- **Symbols:** X ∈ R^(n×d); Q = X W_Q, K = X W_K, V = X W_V; S = QKᵀ ∈ R^(n×n); A = softmax_rows(S); O = AV ∈ R^(n×d_v). A full-width single head uses d_v = d. Here Q/K/V are projected arrays. Row weight matrices are transposes of their column-convention counterparts in C06; the slides sometimes use Q/K/V for the weights themselves.
- **Key points:** This is C06 computed in parallel across positions, fitting GPU execution and allowing direct interactions between tokens. Autoregressive generation still waits for new tokens to be generated (notes §1.2, §3.1; J&M §7.3).
- **Figures:** 1.56.

- *Revised at U7 (#4944):* C12 supplies the scaled production form; the earlier unscaled equations are pedagogical.

### C08 Position information in self-attention
- **Meaning:** Distinguish token positions by adding position-dependent vectors to the inputs or by making attention itself depend on relative positions (slides 43–46; notes §2.2; J&M §7.4).
- **Key points:** Without position information, permuting keys together with their corresponding values preserves their weighted sum for a fixed query. Full unmasked self-attention is permutation-equivariant, not an ordered-sequence encoder. The usual additive input is x̃_i = x_i + p_i, once at the bottom of a deep stack (slide 44).
  - Learned absolute positions use P ∈ R^(N×d), p_i = P_i, with a vector for each supported index. They fit the data but have no learned table entry outside 1…N; outer positions may also get less training data (slide 46; J&M §7.4).
  - Fixed sinusoidal positions use the canonical zero-based equations PE(pos,2r) = sin(pos/10000^(2r/d)), PE(pos,2r+1) = cos(pos/10000^(2r/d)). They require no learned position parameters. Vaswani §3.5 hypothesizes relative-offset learning and length extrapolation; slide 45 cautions that extrapolation does not really work. Being able to compute a longer-index encoding is not demonstrated model generalization.
  - Alammar’s 2020 correction distinguishes Tensor2Tensor’s concatenated sine/cosine halves from the paper’s interleaving. This is an implementation convention within the same method.
  - Relative positions can change attention scores instead of initial absolute input vectors. The notes’ illustrative bias is α_i = softmax([q_iᵀk_j − |i−j|]_j), favoring nearer words when content is held fixed. This is ALiBi-like intuition, not the complete multi-head ALiBi recipe. Slides 46 and 69 name relative linear positions (Shaw) and dependency-syntax positions (Wang). J&M names rotary position embeddings (RoPE) without deriving rotations.
- **Example:** the oven cooked the bread so versus the bread cooked the oven so must have different sequence meanings. Notes Eq. 9 lists six token rows but prints R^(5×d); use six rows, with all six keys in the normalizer (Eq. 11 omits the final so term).
- **Figures:** 1.44, 1.45, 1.46.

### C09 Position-wise feed-forward processing in attention stacks
- **Meaning:** After attention mixes tokens, apply the same nonlinear network separately to each position (slide 48; notes §2.3; J&M §7.2.1).
- **Symbols:** Column form FFN(h_i) = W_2 ReLU(W_1h_i+b_1)+b_2, W_1 ∈ R^(d_ff×d), W_2 ∈ R^(d×d_ff). Parameters are shared across positions within a layer, but differ across layers. The original transformer uses d = 512 and d_ff = 2048 (Vaswani §3.3); the notes’ 5d width is illustrative rather than required.
- **Key points:** The network acts “independently to each word representation” (notes §2.3). Attention mixes positions, while the FFN processes the resulting features with elementwise nonlinearities.
  - Corrected attention-only composition: o_i = Σ_k β_ik V^(2)V^(1)x_k, where β_ik = Σ_j α^(2)_ij α^(1)_jk. Notes Eqs. 15–16 put a j-dependent term outside its sum and misindex the coefficient. This shows repeated value averaging at the realized weights; attention is not globally linear, because its softmax weights depend nonlinearly on the inputs. Marginal note 3 explicitly leaves the sufficiency of attention-only stacks outside its scope.
  - J&M Eq. 7.22 gives a gated row-vector FFN: [Swish_β(xW_1+b_1) ⊙ (xW_2+b_2)]W_3+b_3, with Swish_β(z) = z σ(βz). The extra projection can be balanced with a hidden width below 4d. In the described configurations FFNs hold most parameters and seem to store much factual knowledge; the latter remains a qualified claim, citing Geva and Meng.
- **Example:** The chef who … food in slide 48 passes through shared FF blocks after attention.
- **Extends:** ch03:C06, ch03:C09, ch03:C10, applying feed-forward networks and nonlinear/gated activations inside a transformer.
- **Figures:** 1.48, 3.10.

### C10 Causal masking for self-attention
- **Meaning:** Enforce next-token prediction by allowing the representation at input position i to use only j ≤ i. The representation predicts token i+1, so attending to its own input is allowed (slide 50; notes §2.4; J&M §7.3).
- **Symbols:** M_ij = 0 for j ≤ i and −∞ for j > i; A = softmax_rows(S+M), O = AV. Equivalently, normalize exp(S_ij) over allowed j only.
- **Key points:** “mask out attention to future words” (slide 50). Mask scores before softmax. Zeroing future probabilities from an already normalized full-sequence distribution also requires renormalization of allowed weights. With known training inputs, all positions can be computed simultaneously without seeing their targets; this does not parallelize generation of unknown future tokens.
- **Example:** The triangular mask over [START], The, chef, who includes the diagonal (slide 50); notes Figure 3 shows the corresponding Zuko prefixes.
- **Source note:** Notes Eq. 21’s displayed probability zeroing requires renormalization if its α came from the full sequence. Marginal note 4 recommends a finite negative constant and warns about infinity; this is implementation-dependent numerical advice, not a universal ban on −∞. Its −10^5 example is outside float16’s finite range (maximum magnitude 65504); use a dtype-appropriate masking convention rather than that literal as a general recommendation.
- **Figures:** 1.50, 1.52, 2.3.

### C11 Multi-head attention
- **Meaning:** Run attention with multiple learned query/key/value projections in parallel, concatenate their contextual outputs, and project back to model width (slides 55, 57–58; notes §3.1).
- **Symbols:** Q_l = XW_Ql, K_l = XW_Kl, V_l = XW_Vl; head_l = softmax_rows(Q_lK_lᵀ/√d_k + M)V_l; MultiHead = Concat(head_1,…,head_h)W_O. Omit M for unmasked attention. W_Ql,W_Kl ∈ R^(d×d_k), W_Vl ∈ R^(d×d_v), W_O ∈ R^((h d_v)×d); usual d_k = d_v = d/h. Output has shape n×d. The original example is d = 512, h = 8, d_k = d_v = 64.
- **Key points:** Heads can look in “multiple places … at once” (slide 57), using different subspaces and relevance patterns. Combined n×d projection arrays reshape to n×h×d/h and transpose to h×n×d/h, treating heads like a batch axis. At fixed model width with reduced per-head width, arithmetic is comparable to a full-width single head (slide 58). Explicit attention-weight storage is h×n×n, so this is not a claim of identical weight-storage size.
- **Example:** Slide 55’s hypothetical entity and syntax preferences illustrate distinct focuses without promising that every learned head has a human-named role.
- **Source note:** Notes Eq. 24 must concatenate contextual h_i^(l), not its displayed uncontextualized v_i^(l). Each token’s head output is a d/h-vector, concatenating to a d-vector; the accompanying d×d/h and d×d size claims are incorrect for per-token vectors.
- **Figures:** 1.55, 1.57, 1.58.

### C12 Scaled dot-product attention
- **Meaning:** Divide query–key dot scores by the square root of head width before softmax to control their growth with dimensionality (slide 59; notes §3.4; Vaswani §3.2.1).
- **Symbols:** Attention(Q,K,V) = softmax_rows(QKᵀ/√d_k + M)V, omitting M when unmasked. Under the paper’s illustrative independent zero-mean unit-variance component assumptions, Var(qᵀk) = d_k and the standard deviation is √d_k (Vaswani footnote 4).
- **Key points:** Large logits can put softmax into small-gradient regimes. Scaling addresses the dimensional effect; for multiple heads divide by √d_k, not by the full model width’s square root.
- **Figures:** 1.59.

### C13 Residual connections and the residual stream
- **Meaning:** Preserve a layer input through an identity path and add a learned update: y = x + f(x), with matching widths (slide 61; notes §3.3). The layer learns a change from identity rather than an entirely new mapping.
- **Key points:** The identity branch has Jacobian I; the full derivative is I + J_f, not always 1. The shortcut can support deep optimization. J&M §7.2 follows a d-dimensional residual stream for each token through the stack: sublayers read the stream and add updates. Attention brings information from other token streams, while FFN and norm are position-wise.
  - Add & Norm combines the residual with normalization. Pre-norm: y = x + f(LN(x)); post-norm: y = LN(x + f(x)). Notes §3.3 cites Xiong for better initialization gradients and easier/faster training with pre-norm in the studied settings. J&M’s prenorm stack has a final norm before its LM head (footnote 2). The original Vaswani and Alammar diagrams use post-norm.
- **Example:** Slide 61 contrasts plain and residual layers; its Li et al. ResNet landscape is an illustration, not a theorem about every transformer’s landscape.
- **Extends:** ch04:C21, making its shortcut forward reference concrete.
- **Figures:** 1.61, 3.7, 3.8.

### C14 Layer normalization (and RMSNorm variant)
- **Meaning:** Normalize hidden features within one token’s vector, independently for each token and each example. Statistics are not taken across sequence positions or batch examples (slide 62; notes §3.2).
- **Symbols:** μ_i = (1/d)Σ_r x_ir; v_i = (1/d)Σ_r(x_ir−μ_i)^2; LN(x_i) = γ ⊙ (x_i−μ_i)/√(v_i+ε) + β. Gain γ and bias β are learned d-vectors. The notes omit these affine parameters and ε for exposition.
- **Key points:** Centering/scaling gives approximately zero mean and unit variance before gain/bias; the affine output need not preserve those statistics. Slide 62’s “unit mean” wording should be zero mean. Reducing uninformative activation variation can facilitate training; notes/Xu suggest gradient benefits, not a universal single explanation.
  - Ba §3 contrasts layer norm’s per-example statistics with batch norm’s cross-example statistics. Layer norm uses the same computation at training and test time and imposes no batch-size requirement.
  - J&M §7.2.2 describes RMSNorm as rescaling by root-mean-square without mean-centering. No further RMSNorm derivation is supplied there.
- **Figures:** 1.62.

### C15 Transformer decoder (decoder-only causal LM)
- **Meaning:** Embed the input, add positions, and stack independently parameterized blocks of masked multi-head self-attention and position-wise FFN, with residual/norm around each sublayer; map final states to next-token probabilities (slide 63; notes §3.6).
- **Symbols:** A prenorm block is T = X + MHA_causal(LN(X)), H = T + FFN(LN(T)); J&M adds a final norm after the complete stack. The deck shows postnorm blocks. Each block maps n×d to n×d, enabling stacking.
- **Key points:** Every token state uses its prefix. Layers have the same structure but independent parameters. Notes Figure 4 is decoder-only and has no encoder cross-attention.
- **Example:** GPT-2, GPT-3 and BLOOM are the notes’ source-era examples, not a current-model survey.
- **Extends:** ch01:C40, from the architecture’s name to its composed causal model.
- **Figures:** 1.63, 2.3.

- *Revised at U12 (#4955):* WikiSum’s reported test perplexity/ROUGE-L pairs are 5.04952/12.7 for recurrent seq2seq attention, 2.46645/34.2 for Transformer-ED, 2.22216/33.6 for Transformer-D, and 1.90325/38.8 for the best shown DMCA/MoE configuration (slide 68). L in that results table is input length, not layer count. Variants and lengths change together, so this is not an isolated architecture ablation. J&M Figure 7.24 instead uses L for layer count, d for model width, A for heads, d_ff for FFN width, and |V| for vocabulary. Its August 2026 configuration examples range from GPT-2 XL (1.5B, 1K context) to Llama 3.1 405B (128K context); the full source-era table is Figure 3.24 here. Figures 1.68, 3.24.

### C16 Transformer encoder
- **Meaning:** Stack the attention/FFN/residual/norm ingredients without a future mask, so each input token can use both left and right context. This builds contextual representations rather than an autoregressive generator (slide 64; notes §3.5).
- **Key points:** Embed and add positions at the bottom, then apply independently parameterized layers. A linear-plus-softmax head can supply token labels or probabilities. The notes name BERT’s masked-token objective and defer its treatment.
- **Figures:** 1.64.

### C17 Transformer encoder–decoder and projected cross-attention
- **Meaning:** Encode the complete source bidirectionally, then generate the target causally. Each decoder block inserts cross-attention between its masked self-attention and FFN (slides 65–66; notes §3.7).
- **Symbols:** Decoder states Z ∈ R^(m×d), final encoder outputs H_enc ∈ R^(n×d); Q = ZW_Q, K = H_enc W_K, V = H_enc W_V. Per-head output = softmax_rows(QKᵀ/√d_k)V, shape m×d_v before head combination. Source and target lengths n and m need not match; slide 66 reuses n illustratively.
- **Key points:** All decoder cross-attention layers read the final encoder output, each with its own projections. The original Vaswani example uses six encoder and six decoder layers; this is a setting, not the definition. Notes §3.7 reports advantages over decoder-only at modest scale, while noting the split of parameters and prevalence of decoder-only at the largest source-era scales.
- **Example:** Translation and article summarization require full source context and a generated target.
- **Extends:** ch04:C25, replacing recurrent encoding/target generation with transformer stacks while retaining conditional seq2seq. C05 is the earlier recurrent cross-attention comparison.
- **Figures:** 1.65, 1.66.

- *Revised at U12 (#4955):* WMT14 evidence comes from Vaswani Table 2/abstract: base BLEU EN-DE 27.3, EN-FR 38.1 with estimated training cost 3.3×10^18 FLOPs; big 28.4/41.8 with cost 2.3×10^19. Slide 67 shows prior-system comparison rows but omits the transformer rows. The paper’s §6.1 prose says 41.0 EN-FR where its table/abstract say 41.8; this internal discrepancy remains explicitly recorded. Results and cost estimates apply to those tasks/configurations, not all-task superiority. Figures 1.67, 4.2.

### C18 Language-modeling head and tied unembedding
- **Meaning:** Convert a final contextual vector into vocabulary logits and next-token probabilities (J&M §7.5).
- **Symbols:** u_i = h_i^L U, y_i = softmax(u_i), U ∈ R^(d×|V|). With row input embeddings E ∈ R^(|V|×d), weight tying sets U = Eᵀ. A separately learned output matrix is also possible.
- **Key points:** Tied weights are optimized for both input lookup and output scoring. Training applies the head at every known position; generation uses the newest position (J&M footnote 3). This vocabulary-output head is distinct from an attention head.
- **Example:** Thanks for all the gives a distribution containing fish (J&M Figure 7.15).
- **Extends:** ch04:C11, retaining the prediction role with transformer states and explicitly tied input/output weights.
- **Figures:** 3.15, 3.16.

### C19 Self-supervised transformer pretraining and teacher forcing
- **Meaning:** Learn all model weights from raw token sequences by predicting each true next token. Teacher forcing supplies the true prefix instead of generated guesses (J&M §7.7).
- **Symbols:** L = −(1/T)Σ_t ln p_θ(w_(t+1) given w_≤t), averaged over T target positions. Each observed input position predicts the following token.
- **Key points:** The causal mask enables all known positions to provide targets in one pass without exposing those targets to their predictors. Gradients update all weights, including embeddings. J&M describes web text plus other quality data, large token batches and shorter documents packed with end-of-text separators. This raw-text self-supervision differs from MT’s paired source/target supervision.
  - The original Vaswani §5 recipe uses Adam, 4000-step linear warmup then inverse-square-root decay: η = d^−0.5 min(s^−0.5, s·warmup^−1.5). It applies dropout to sublayer outputs and embedding-plus-position sums (base rate 0.1), and label smoothing 0.1 (softened targets rather than one-hot labels). The paper reports that smoothing worsens perplexity while improving accuracy/BLEU. These are source-specific settings, not defining requirements. Original embedding vectors are scaled by √d (Vaswani §3.4).
- **Example:** So long and thanks for … supplies shifted targets long and thanks for all … (J&M Figure 7.22).
- **Extends:** ch01:C39, ch04:C14, adding parallel transformer training and explicit shifted targets.
- **Figures:** 3.22.

- *Revised at U14 (#4959):* J&M Historical Notes distinguishes pretrained static embeddings (LSA/SVD and word2vec) reused in tasks from a whole pretrained contextual LM reused frozen or adapted. Bengio joins learned embeddings with next-word prediction; RNN LMs, word2vec and Dai/Le–ELMo-style model transfer lead toward GPT-2 zero-shot applications. This refines ch01:C39 and ch02:C36 rather than naming a second pretraining concept. J&M’s simplified ELMo story is its historical account, not a claim that every task in the original work used one universal fine-tuning pipeline.

### C20 Perplexity for transformer LM evaluation
- **Meaning:** Normalized inverse held-out sequence likelihood, measuring next-token predictive accuracy (J&M §7.7.1).
- **Symbols:** PPL = exp[−(1/T)Σ_t ln p_θ(w_t given w_<t)] = P(w_1:T)^(−1/T). With natural logarithms, average cross-entropy is measured in nats.
- **Key points:** Lower perplexity means higher normalized likelihood on the evaluated data. It is not a general task-quality score. Tokenization changes the number and identity of prediction events, so exact comparisons should use the same tokenizer (J&M §7.7.1).
- **Extends:** ch04:C16, adding the tokenizer caveat and transformer context.
- **Figures:** 3.27.

### C21 Autoregressive decoding: greedy choice, sampling and beam search
- **Meaning:** Choose a next token from LM probabilities, append it and repeat with the new context until EOS or a length limit (J&M §7.6).
- **Key points:** Greedy choice is argmax_w p(w given prefix): locally most probable, without guaranteeing the best whole sequence. It is deterministic for a fixed model and context. Random multinomial sampling draws w from that distribution, making probable tokens more likely while allowing rare ones. J&M contrasts generic/repetitive greedy output with unrestricted sampling’s aggregate low-probability tail.
  - Alammar’s decoding discussion introduces beam search: retain a bounded set of partial hypotheses, expand them and prune by sequence scores. Beam size two illustrates search breadth, without guaranteeing the globally best output. J&M defers fuller translation-oriented beam search to Chapter 13.
- **Example:** After So long and thanks for, the toy distribution is all .44, the .33, your .15, that .08. Greedy picks all; Figure 7.19 illustrates sampling the (J&M Figures 7.18–7.19).
- **Extends:** ch04:C07, adding explicit decoding policies and transformer generation.
- **Figures:** 3.18, 3.19.

- *Revised at U14 (#4959):* J&M footnote 4 uses autoregressive loosely for a nonlinear model conditioned on preceding tokens; it does not assert technical linear autoregression of the transformer.

### C22 Temperature sampling
- **Meaning:** Reshape probabilities by scaling logits before softmax, then sample (J&M §7.6.3).
- **Symbols:** p_τ(w) = exp(u_w/τ)/Σ_v exp(u_v/τ), τ > 0. At τ = 1 the distribution is unchanged; τ < 1 sharpens it and τ > 1 flattens it. As τ approaches 0, mass concentrates on maximal logits (uniform over tied maxima); as τ grows without bound it approaches uniform over a finite unmasked vocabulary.
- **Key points:** Scaling changes logit differences; negative logits need not become numerically larger for sharpening to occur. The source’s quality/diversity discussion describes tendencies, not guarantees of factuality.
- **Example:** In Figure 7.21, all rises from .44 at τ = 1 to .59 at .5 and .95 at .1; τ = 100 gives a nearly uniform toy distribution.
- **Figures:** 3.21.

### C23 Top-k and nucleus (top-p) sampling
- **Meaning:** Remove low-probability candidates, renormalize the remainder and sample (J&M §7.6.4).
- **Symbols:** Top-k keeps k most probable tokens; k = 1 gives greedy choice. Top-p keeps the smallest probability-sorted prefix S whose total probability is at least p, 0 < p ≤ 1. Its candidate count adapts to distribution concentration rather than retaining a fixed percentage of vocabulary entries.
- **Key points:** Both truncate the tail. J&M describes combining top-k followed by top-p. This changes probability support, unlike temperature scaling of finite logits.
- **Figures:** 3.21.

### C24 Dense attention’s compute, storage and long-context trade-off
- **Meaning:** All-pairs interaction gives direct communication and parallel token computation, with costs quadratic in sequence length (slides 69–70; Vaswani §4).
- **Symbols:** At width d, pairwise score/value operations cost O(n²d); explicitly materialized weights occupy O(hn²). Projections also cost roughly O(nd²), and FFNs O(nd·d_ff), distinct from the pairwise term.
- **Key points:** Vaswani Table 1 contrasts recurrent O(n) sequential depth/path length with dense attention O(1); its attention complexity comparison omits separately described position-wise processing. The compute advantage depends on n versus d. Causal masking alone does not remove quadratic scaling. Slide 70’s observations about compute outside attention and cheaper alternatives are the lecture’s source-era claims and research question, not a present-day universal survey.
- **Example:** Slide 69 asks about n ≥ 50,000; slide 70 asks what better methods might enable beyond 100k tokens.
- **Figures:** 1.69, 1.70, 4.1.

### C25 KV caching for autoregressive inference
- **Meaning:** Reuse each layer’s previously computed keys and values as new tokens arrive rather than recomputing the prefix (J&M §7.8).
- **Key points:** Compute the new query/key/value, append the new K/V and attend over retained keys/values. Causal past states do not depend on future tokens. Reuse saves prefix work but still requires attending over a growing prefix and storing cached vectors; it does not make total dense-attention generation work linear.
- **Example:** J&M Figure 7.23 computes q4 against cached k1…k3 and new k4, with corresponding cached/new values.
- **Figures:** 3.23.

### C26 Image Transformer: autoregressive images with local attention
- **Meaning:** Serialize pixel channels/values and model log p(x) = Σ_t log p(x_t given x_<t), using transformer blocks and causal local neighborhoods (Parmar §§3–6).
- **Key points:** Half of each position encoding represents row and half column/channel; learned and sinusoidal forms are compared. Query blocks share a larger memory block, preserving batched attention. One-dimensional locality follows raster order; two-dimensional locality uses rectangular blocks with more balanced vertical/horizontal context. Ungenerated positions are masked in both (Figures 1–2). With N positions and fixed memory size r, attention costs O(Nrd); learned parameter count is decoupled from receptive-field size.
  - Output heads use either a 256-way categorical distribution per RGB channel or a discretized logistic mixture over whole pixels. The mixture captures intensity ordering/channel dependence and uses fewer output parameters (§3.4).
  - Unconditional and class-conditioned generation use a decoder-only model, with class embeddings supplying conditioning. Super-resolution encodes an 8×8 image and locally decodes a 32×32 output. Plausible details are synthesized rather than recovered ground truth. Temperature controls sampling (§4).
- **Example/evidence:** Reported ImageNet NLL is 3.77 bits/dimension versus the cited Gated PixelCNN’s 3.83. In the specified human study, generated images were chosen as real about 36% of the time versus about 11% for PixelRecursive. These are dataset/protocol-specific results. Section 5.3/Table 5 favors τ = .8, while Table 6’s caption says .9; the inconsistency remains recorded.
- **Figures:** 5.1, 5.2, 5.5, 5.6.

### C27 Music Transformer: relative attention for symbolic musical structure
- **Meaning:** Autoregressively model serialized symbolic music with relative-position attention, supporting repeats, motifs and longer structure (Huang §§3–5 and appendices).
- **Symbols:** Relative logits S_rel(i,j) = q_iᵀe_(j−i); attention = softmax[(QKᵀ+S_rel)/√d_h]V. JSB Chorales serializes four voices on a sixteenth-note grid. Piano-e-Competition uses NOTE_ON/OFF, TIME_SHIFT and VELOCITY events (128/128/100/32 token types), so sequence distance is not literally elapsed time.
- **Key points:** Compute Q E_relᵀ, then skew its relative-offset columns into key-position columns, avoiding an L×L×d_h gathered tensor. Global skew: pad a left column, reshape to (L+1,L), keep the final L rows (§3.4, Figure 1). The relative-embedding/intermediate term falls from O(L²D) to O(LD), but logits still require O(L²). Overall relative storage is O(LD+L²) and time remains O(L²D) (Table 1).
  - A local version attends within and before blocks. Content-relative timing/pitch additions (§4.1.1) again require O(L²D) gathered tensors; they are outside the efficient sequence-offset claim.
- **Example/evidence:** Elaborate a Chopin motif or generate a performance/accompaniment conditioned on melody. Piano-e-Competition validation NLL is 1.835 versus baseline Transformer 1.861; melody-conditioned NLL is 1.786 versus 2.066. Qualitative samples extend beyond training length. The listening test favors relative Transformer over baseline Transformer, but does not establish significant head-to-head superiority over LSTM at the corrected threshold. Short clips do not establish universal long-form quality. Appendix C visualizes recurring high-note and left-hand patterns.
- **Figures:** 6.1, 6.2, 6.7, 6.8.

### C28 Maximum-entropy language modeling and trigger features
- **Meaning:** A historical logistic-regression word predictor combines n-gram information with more distant contextual features, including associated-word trigger pairs, and learns to predict following words from running text (J&M Historical Notes).
- **Key points:** Rosenfeld’s 1994/1996 model is a “logistic regression predictor” and a precursor of self-supervised next-word learning, not a neural embedding model. This is a distinct model beyond the counted n-gram models in ch04:C04 and the general statistical-modeling history in ch01:C26.
- **Example:** N-gram features, distant n-grams and trigger-pair word associations combine immediate and longer context. The thesis abstract confirms the motivation of adapting word expectations to the topic of discourse.

## Problems

What each end-of-chapter problem asks and which concepts it exercises. Record no solutions the source doesn't give.

| Problem | Asks | Exercises |
|---|---|---|
| J&M 7.1 | Count attention projection and FFN parameters, inspect dependence on head count, and show the requested approximate stack count, ignoring bias/norm. | C11, C09, C15 |
| J&M 7.2 | Explain why the score mask uses −∞ rather than 0 and what 0 would do. | C10 |
| J&M 7.3 | Compute perplexity for It was a dream with successive probabilities .0001, .0002, .000005 and .0000001. | C20 |
| J&M 7.4 | Prove that perplexity equals exp of mean cross-entropy. | C20 |
| J&M 7.5 | Use logits 1.2, .9, .1 and −.5 to calculate distributions at τ = .2 and 2, verify normalization and examine the 0/infinity limits. | C22 |

No solutions are added. Note 10 §3.1’s historical Assignment 5 reference is an intuition pointer, not a Winter 2026 assigned exercise.

## Forward references

Topics the chapter defers. None is taught here; later extraction EXTENDs from these hooks.

| Topic | Chapter | Claim / promised treatment |
|---|---|---|
| Attention-only stack sufficiency | Beyond notes §2.3 | Marginal note 3 says the answer is nuanced and outside the note’s scope. |
| Masked/bidirectional language modeling and BERT | J&M Chapter 9; notes §3.5 | Encoder representations support masked-token prediction; its objective is deferred. |
| Post-training, instruction tuning and alignment | J&M Chapter 8 | The reading defers adapting/aligning pretrained models. |
| Translation and fuller beam search | J&M Chapter 13 | The reading points to constrained translation generation. |
| Speech recognition | J&M Chapter 16 | Apply transformer/encoder–decoder ideas to speech. |
| Efficient long-context attention and richer position schemes | Slides 69–70 | Quadratic costs and alternatives to simple absolute positions remain research questions. |
| MoE, GQA and quantization | Future J&M coverage | Named efficiency innovations, without derivations in this chapter. |
| Text-conditioned images, video and model-based reinforcement learning | Image Transformer conclusion | Proposed research directions, not results established by this reading. |

## Studs

Concepts the chapter uses but does not teach. A stud means "not taught here", not "nothing more to learn".


### B01 RNN hidden-state recurrence
- **Meaning:** The same state-update rule processes successive tokens.
- **Used at:** notes §1.2.
- **Status:** Taught in ch04:C10.

### B02 Embedding lookup and static word vectors
- **Meaning:** A vocabulary index or one-hot token selects its learned dense vector.
- **Used at:** notes §1.1.
- **Status:** Taught in ch03:C04, ch02:C07.

### B03 Matrix/vector operations and dependency paths
- **Meaning:** Matrix products, sums and paths of operations determine how vectors can be computed.
- **Used at:** notes §1.2.
- **Status:** not taught here; further prerequisites are not assessed in this lecture.

### B04 Chain rule and vanishing/exploding gradients
- **Meaning:** Gradients compose along computation paths and can attenuate or grow.
- **Used at:** slides 3–13.
- **Status:** Taught in ch03:C16, ch04:C17, ch04:C19.

### B05 Recurrent seq2seq conditional LM and bottleneck
- **Meaning:** An encoder’s fixed vector conditions a target-generating decoder, trained end to end.
- **Used at:** slides 14–21.
- **Status:** Taught in ch04:C25, ch04:C26.

### B06 Softmax
- **Meaning:** Exponentiated scores normalized over the specified alternatives form a distribution.
- **Used at:** notes §1.1, Eq. 2.
- **Status:** Taught in ch02:C13.

### B07 Sine, cosine and periodicity
- **Meaning:** Periodic trigonometric functions can encode position at multiple frequencies.
- **Used at:** slide 45; Vaswani §3.5.
- **Status:** not taught here; further prerequisites are not assessed in this lecture.

### B08 Feed-forward networks and elementwise nonlinear/gated activations
- **Meaning:** Learned affine transformations combine with elementwise activations and gates to process features.
- **Used at:** slide 48; notes §2.3; J&M §7.2.1.
- **Status:** Taught in ch03:C06, ch03:C07, ch03:C08, ch03:C09, ch03:C10.

### B09 Autoregressive next-token modeling and conditional language models
- **Meaning:** Predict each next token from an allowed preceding context, with a source sequence as optional conditioning information.
- **Used at:** notes §2.4; J&M §7.3.
- **Status:** Taught in ch04:C01, ch04:C11, ch04:C26.

### B10 Mean, variance and standard deviation
- **Meaning:** Mean measures center, variance mean squared deviation, and standard deviation its square root.
- **Used at:** Vaswani §3.2.1, footnote 4; slide 62.
- **Status:** not taught here; further prerequisites are not assessed in this lecture.

### B11 Cross-entropy and gradient-based training
- **Meaning:** Compare predicted distributions with targets and update parameters using backpropagation and gradient-based optimization.
- **Used at:** J&M §7.7; Vaswani §5. The original recipe additionally uses dropout, Adam and a learning-rate schedule.
- **Status:** Taught in ch03:C11, ch03:C24, ch02:C16. Recipe-specific background: Taught in ch03:C32, ch03:C35, ch03:C37.

### B12 Logarithms, exponentials and conditional-probability products
- **Meaning:** Products of conditional probabilities assign sequence likelihoods, and logs/exponentials convert products and averages into loss/perplexity forms.
- **Used at:** J&M §7.7.1.
- **Status:** not taught here; further prerequisites are not assessed in this lecture.

### B13 Categorical sampling and argmax
- **Meaning:** Sampling chooses according to probabilities; argmax selects an item with a largest score.
- **Used at:** J&M §7.6.1–7.6.2.
- **Status:** not taught here; further prerequisites are not assessed in this lecture.

### B14 BLEU
- **Meaning:** A translation-overlap metric based on n-gram matches, with a brevity penalty.
- **Used at:** slide 67; Vaswani Table 2.
- **Status:** not taught here; further prerequisites are not assessed in this lecture.

### B15 ROUGE-L
- **Meaning:** A reference-overlap metric based on longest common subsequence.
- **Used at:** slide 68’s WikiSum comparison.
- **Status:** not taught here; further prerequisites are not assessed in this lecture.

### B16 Asymptotic complexity and tensor-storage size
- **Meaning:** Big-O describes how computation or storage grows with sequence length and model dimensions.
- **Used at:** slides 69–70; Vaswani Table 1.
- **Status:** not taught here; further prerequisites are not assessed in this lecture.

### B17 RGB pixel intensities and coordinates
- **Meaning:** Images have spatial pixel positions and color-channel intensities.
- **Used at:** Parmar §3.1.
- **Status:** not taught here; further prerequisites are not assessed in this lecture.

### B18 Symbolic music: notes, voices, timing and velocity
- **Meaning:** A score/performance can represent pitches and voices with note on/off, timing and intensity events.
- **Used at:** Huang §3.1 and Appendix A.
- **Status:** not taught here; further prerequisites are not assessed in this lecture.

### B19 Mixture distributions
- **Meaning:** A weighted combination of component probability distributions provides a richer output model.
- **Used at:** Parmar §3.4’s optional discretized logistic-mixture output head.
- **Status:** not taught here; further prerequisites are not assessed in this lecture.

### B20 Logistic/maximum-entropy statistical modeling
- **Meaning:** A family of conditional statistical predictors using weighted contextual features.
- **Used at:** J&M Historical Notes names the family without deriving its training principle.
- **Status:** not taught here; further prerequisites are not assessed in this lecture.

## Prerequisites

Every row reads "From `prerequisite_for` To"; a learner walks backward from To. The prerequisite rows must form
no cycle. A **scope** means the prerequisite is needed only for that part of To.

| From | Predicate | To | Family | Status | Scope | Basis |
|---|---|---|---|---|---|---|
| B02 | `prerequisite_for` | C01 | prerequisite | inferred | — | notes §1.1: contextual representations extend embedding lookup |
| B01 | `prerequisite_for` | C02 | prerequisite | inferred | — | notes §1.2: expansion of the recurrent update |
| B03 | `prerequisite_for` | C02 | prerequisite | inferred | — | notes §1.2: matrix-operation dependencies and path length |
| B04 | `prerequisite_for` | C02 | prerequisite | inferred | gradient-path motivation only | slides 3–13; notes §1.2’s learning difficulty |
| B01 | `prerequisite_for` | C03 | prerequisite | inferred | — | slide 23 pools RNN hidden states |
| B03 | `prerequisite_for` | C03 | prerequisite | inferred | — | slide 23’s elementwise mean/max |
| B03 | `prerequisite_for` | C04 | prerequisite | inferred | — | slides 24, 38: weighted vector sums |
| B06 | `prerequisite_for` | C04 | prerequisite | inferred | normalized attention weights | slides 24, 29: softmax over scores |
| C04 | `prerequisite_for` | C05 | prerequisite | inferred | — | slides 25–31 apply soft lookup to seq2seq |
| B05 | `prerequisite_for` | C05 | prerequisite | inferred | — | slide 22 answers the encoder–decoder bottleneck |
| B04 | `prerequisite_for` | C05 | prerequisite | inferred | gradient-shortcut claim only | slide 37: shorter paths to faraway states |
| C04 | `prerequisite_for` | C06 | prerequisite | inferred | — | notes §2.1 specializes soft lookup to one sequence |
| B02 | `prerequisite_for` | C06 | prerequisite | inferred | — | slide 42: project input embeddings |
| C01 | `prerequisite_for` | C06 | prerequisite | inferred | interpreting contextual output | notes §2.1; J&M §7.1 |
| C06 | `prerequisite_for` | C07 | prerequisite | inferred | — | slide 56 computes scalar attention in matrices |
| B03 | `prerequisite_for` | C07 | prerequisite | inferred | — | slide 56: matrix dimensions and multiplication |
| C02 | `prerequisite_for` | C07 | prerequisite | inferred | recurrence comparison only | notes §1.2, §3.1 |
| C06 | `prerequisite_for` | C08 | prerequisite | inferred | — | notes §2.2: identify the order information absent from raw attention |
| B02 | `prerequisite_for` | C08 | prerequisite | inferred | additive position vectors | slide 44 adds vectors to token embeddings |
| B07 | `prerequisite_for` | C08 | prerequisite | inferred | sinusoidal equations only | slide 45; Vaswani §3.5 |
| C06 | `prerequisite_for` | C09 | prerequisite | inferred | role in attention stacks | slide 48 processes attention output |
| B08 | `prerequisite_for` | C09 | prerequisite | inferred | — | notes Eq. 17; J&M Eq. 7.22 |
| C07 | `prerequisite_for` | C10 | prerequisite | inferred | — | slide 50 masks the score matrix |
| B09 | `prerequisite_for` | C10 | prerequisite | inferred | — | notes §2.4: predicting a token without its future |
| C07 | `prerequisite_for` | C11 | prerequisite | inferred | — | slide 57 runs matrix attention per head |
| B03 | `prerequisite_for` | C11 | prerequisite | inferred | dimensions and head batching | slide 58 |
| C10 | `prerequisite_for` | C11 | prerequisite | inferred | masked variant only | notes §3.1 with §2.4 |
| C06 | `prerequisite_for` | C12 | prerequisite | inferred | — | slide 59 modifies dot-product scores |
| B06 | `prerequisite_for` | C12 | prerequisite | inferred | softmax effect | Vaswani §3.2.1 |
| B10 | `prerequisite_for` | C12 | prerequisite | inferred | variance argument only | Vaswani footnote 4 |
| B03 | `prerequisite_for` | C13 | prerequisite | inferred | matching-width sums | slide 61 |
| B04 | `prerequisite_for` | C13 | prerequisite | inferred | gradient reasoning only | notes §3.3 |
| C14 | `prerequisite_for` | C13 | prerequisite | inferred | Add & Norm only | notes §3.3, Eqs. 30–31 |
| C06 | `prerequisite_for` | C13 | prerequisite | inferred | attention’s residual-stream role only | J&M §7.2 |
| C09 | `prerequisite_for` | C13 | prerequisite | inferred | FFN’s residual-stream role only | J&M §7.2 |
| B03 | `prerequisite_for` | C14 | prerequisite | inferred | — | slide 62: per-token vector operations |
| B10 | `prerequisite_for` | C14 | prerequisite | inferred | — | notes §3.2: mean and variance |
| C08 | `prerequisite_for` | C15 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C09 | `prerequisite_for` | C15 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C10 | `prerequisite_for` | C15 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C11 | `prerequisite_for` | C15 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C13 | `prerequisite_for` | C15 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C14 | `prerequisite_for` | C15 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C08 | `prerequisite_for` | C16 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C09 | `prerequisite_for` | C16 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C11 | `prerequisite_for` | C16 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C13 | `prerequisite_for` | C16 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C14 | `prerequisite_for` | C16 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C01 | `prerequisite_for` | C16 | prerequisite | inferred | interpreting representation output | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C15 | `prerequisite_for` | C17 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C16 | `prerequisite_for` | C17 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C04 | `prerequisite_for` | C17 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| B05 | `prerequisite_for` | C17 | prerequisite | inferred | — | slides 63–66; notes §3.5–3.7 compose the stated ingredients |
| C01 | `prerequisite_for` | C18 | prerequisite | inferred | — | J&M §7.5, §7.7: output, training and evaluation definitions |
| B02 | `prerequisite_for` | C18 | prerequisite | inferred | — | J&M §7.5, §7.7: output, training and evaluation definitions |
| B06 | `prerequisite_for` | C18 | prerequisite | inferred | — | J&M §7.5, §7.7: output, training and evaluation definitions |
| C15 | `prerequisite_for` | C18 | prerequisite | inferred | placement on decoder only | J&M §7.5, §7.7: output, training and evaluation definitions |
| C15 | `prerequisite_for` | C19 | prerequisite | inferred | — | J&M §7.5, §7.7: output, training and evaluation definitions |
| C18 | `prerequisite_for` | C19 | prerequisite | inferred | — | J&M §7.5, §7.7: output, training and evaluation definitions |
| B11 | `prerequisite_for` | C19 | prerequisite | inferred | — | J&M §7.5, §7.7: output, training and evaluation definitions |
| B09 | `prerequisite_for` | C20 | prerequisite | inferred | — | J&M §7.5, §7.7: output, training and evaluation definitions |
| B11 | `prerequisite_for` | C20 | prerequisite | inferred | cross-entropy equivalence only | J&M §7.5, §7.7: output, training and evaluation definitions |
| B12 | `prerequisite_for` | C20 | prerequisite | inferred | — | J&M §7.5, §7.7: output, training and evaluation definitions |
| C18 | `prerequisite_for` | C21 | prerequisite | inferred | — | J&M §7.6: next-token distributions and policy operations |
| B09 | `prerequisite_for` | C21 | prerequisite | inferred | — | J&M §7.6: next-token distributions and policy operations |
| B13 | `prerequisite_for` | C21 | prerequisite | inferred | — | J&M §7.6: next-token distributions and policy operations |
| C18 | `prerequisite_for` | C22 | prerequisite | inferred | — | J&M §7.6: next-token distributions and policy operations |
| C21 | `prerequisite_for` | C22 | prerequisite | inferred | — | J&M §7.6: next-token distributions and policy operations |
| B12 | `prerequisite_for` | C22 | prerequisite | inferred | — | J&M §7.6: next-token distributions and policy operations |
| C21 | `prerequisite_for` | C23 | prerequisite | inferred | — | J&M §7.6: next-token distributions and policy operations |
| C07 | `prerequisite_for` | C24 | prerequisite | inferred | — | slides 67–70; J&M §7.8; reported operation or evidence |
| B16 | `prerequisite_for` | C24 | prerequisite | inferred | — | slides 67–70; J&M §7.8; reported operation or evidence |
| C02 | `prerequisite_for` | C24 | prerequisite | inferred | recurrence comparison only | slides 67–70; J&M §7.8; reported operation or evidence |
| C15 | `prerequisite_for` | C25 | prerequisite | inferred | — | slides 67–70; J&M §7.8; reported operation or evidence |
| C07 | `prerequisite_for` | C25 | prerequisite | inferred | — | slides 67–70; J&M §7.8; reported operation or evidence |
| C21 | `prerequisite_for` | C25 | prerequisite | inferred | — | slides 67–70; J&M §7.8; reported operation or evidence |
| B14 | `prerequisite_for` | C17 | prerequisite | inferred | MT evidence only | slides 67–70; J&M §7.8; reported operation or evidence |
| B15 | `prerequisite_for` | C15 | prerequisite | inferred | WikiSum evidence only | slides 67–70; J&M §7.8; reported operation or evidence |
| C20 | `prerequisite_for` | C15 | prerequisite | inferred | WikiSum perplexity only | slides 67–70; J&M §7.8; reported operation or evidence |
| C15 | `prerequisite_for` | C26 | prerequisite | inferred | — | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| C08 | `prerequisite_for` | C26 | prerequisite | inferred | — | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| C24 | `prerequisite_for` | C26 | prerequisite | inferred | locality/complexity trade-off | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| B17 | `prerequisite_for` | C26 | prerequisite | inferred | — | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| C17 | `prerequisite_for` | C26 | prerequisite | inferred | super-resolution only | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| C22 | `prerequisite_for` | C26 | prerequisite | inferred | temperature sampling only | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| B19 | `prerequisite_for` | C26 | prerequisite | inferred | logistic-mixture output only | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| C15 | `prerequisite_for` | C27 | prerequisite | inferred | — | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| C11 | `prerequisite_for` | C27 | prerequisite | inferred | — | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| C08 | `prerequisite_for` | C27 | prerequisite | inferred | — | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| C24 | `prerequisite_for` | C27 | prerequisite | inferred | complexity comparison | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| B18 | `prerequisite_for` | C27 | prerequisite | inferred | — | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| C17 | `prerequisite_for` | C27 | prerequisite | inferred | melody conditioning only | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| B03 | `prerequisite_for` | C27 | prerequisite | inferred | skew indexing | Parmar §3–4; Huang §3–4: adaptation, representation or computation |
| B09 | `prerequisite_for` | C28 | prerequisite | inferred | — | J&M Historical Notes: predict the next word from its context |
| B20 | `prerequisite_for` | C28 | prerequisite | inferred | — | J&M Historical Notes: logistic/maxent predictor |
| B02 | `prerequisite_for` | C19 | prerequisite | inferred | embedding versus whole-model transfer distinction | J&M Historical Notes |

### Per-concept review

| Concept | Direct prerequisites and notes |
|---|---|
| C01 | B02: static embeddings supply the representation being contextualized. |
| C02 | B01, B03; B04 for gradient-path motivation only. |
| C03 | B01, B03; pooling is a baseline, not a prerequisite of attention. |
| C04 | B03, B06. |
| C05 | C04, B05; B04 for the gradient-shortcut claim only. |
| C06 | C04, B02; C01 for interpreting the contextual result. |
| C07 | C06, B03; C02 for the recurrence comparison only. |
| C08 | C06; B02 for additive positions; B07 for sinusoidal equations. Raw self-attention does not require position encoding to be defined. |
| C09 | C06 for its role in the stack; B08 for the network and activations. |
| C10 | C07, B09. Index-based masking does not require learned position vectors. |
| C11 | C07, B03; C10 for the masked variant. Multi-head composition does not require scaling to be defined. |
| C12 | C06, B06; B10 for the variance argument only. |
| C13 | B03; B04 for gradients, C14 for Add & Norm, C06/C09 for residual-stream sublayer roles. |
| C14 | B03, B10. Statistics are per token and per example. |
| C15 | C08, C09, C10, C11, C13, C14; B15 and C20 for WikiSum evidence only. |
| C16 | C08, C09, C11, C13, C14, C01. C01 is scoped to interpreting the representation output. |
| C17 | C15, C16, C04, B05; B14 for MT evidence only. |
| C18 | C01, B02, B06; C15 for placement on a transformer decoder. |
| C19 | C15, C18, B11; B11’s optimizer/dropout background is scoped to the original recipe, B02 to the embedding/model-transfer distinction. |
| C20 | B09, B12; B11 for the cross-entropy equivalence. |
| C21 | C18, B09, B13. |
| C22 | C18, C21, B12. |
| C23 | C21; normalized distributions already require B06 through C18. |
| C24 | C07, B16; C02 for recurrence comparison. |
| C25 | C15, C07, C21; reuse does not require C24 to be understood. |
| C26 | C15, C08, B17; C24 for locality/cost, C17 for super-resolution, C22 for temperature, B19 for logistic mixtures. |
| C27 | C15, C11, C08, B18; C24 for complexity, B03 for skew indexing, C17 for melody conditioning. |
| C28 | B09, B20. The chapter supplies the historical feature combination, not a derivation of maxent estimation. |

## Predicates

The controlled list is `predicates.md` in this pack. List here only predicates this map ADDED with the
Reviewer's agreement, each with its family and meaning.

## Links

Functional links from the text, each marked **stated** (the author says it) or **inferred** (our reading, with its reason).

| From | Predicate | To | Family | Status | Scope / original wording | Basis |
|---|---|---|---|---|---|---|
| C05 | `uses` | C04 | functional | stated | per-step source lookup | slides 25–31 |
| C04 | `contrasts_with` | C03 | functional | stated | query-dependent weighting versus fixed pooling | slides 23–24 |
| C06 | `kind_of` | C04 | taxonomic | stated | same-sequence soft lookup | notes §2–2.1 |
| C06 | `produces` | C01 | functional | stated | contextual token vector | notes §2.1 |
| C06 | `contrasts_with` | C05 | functional | stated | same-sequence versus cross-sequence origins | slide 40 |
| C07 | `computes` | C06 | functional | stated | sequence matrix form | slide 56; notes §3.1 |
| C07 | `contrasts_with` | C02 | functional | inferred | parallel within-layer positions versus recurrence | notes §1.2 and §3.1 |
| C08 | `applies_to` | C06 | functional | stated | supply missing order information | slides 43–46; notes §2.2 |
| C10 | `applies_to` | C06 | functional | stated | causal self-attention | slide 50 |
| C11 | `uses` | C06 | functional | stated | repeated attention with separate projections | slide 57 |
| C12 | `applies_to` | C06 | functional | stated | scale query–key scores | slide 59 |
| C11 | `uses` | C12 | functional | stated | scaled per-head attention | slide 59 |
| C13 | `uses` | C14 | functional | stated | Add & Norm only | notes §3.3 |
| C15 | `has_part` | C11 | part-whole | stated | composed block ingredient | slides 63–64; notes §3.5–3.6 |
| C15 | `has_part` | C09 | part-whole | stated | composed block ingredient | slides 63–64; notes §3.5–3.6 |
| C15 | `has_part` | C13 | part-whole | stated | composed block ingredient | slides 63–64; notes §3.5–3.6 |
| C15 | `has_part` | C14 | part-whole | stated | composed block ingredient | slides 63–64; notes §3.5–3.6 |
| C15 | `uses` | C08 | functional | stated | input positions or causal constraint | slides 63–64 |
| C15 | `uses` | C10 | functional | stated | input positions or causal constraint | slides 63–64 |
| C16 | `has_part` | C11 | part-whole | stated | composed block ingredient | slides 63–64; notes §3.5–3.6 |
| C16 | `has_part` | C09 | part-whole | stated | composed block ingredient | slides 63–64; notes §3.5–3.6 |
| C16 | `has_part` | C13 | part-whole | stated | composed block ingredient | slides 63–64; notes §3.5–3.6 |
| C16 | `has_part` | C14 | part-whole | stated | composed block ingredient | slides 63–64; notes §3.5–3.6 |
| C16 | `uses` | C08 | functional | stated | input positions or causal constraint | slides 63–64 |
| C16 | `contrasts_with` | C15 | functional | stated | bidirectional versus causal context | slide 64 |
| C17 | `has_part` | C16 | part-whole | stated | complete-source encoder | slide 65 |
| C17 | `has_part` | C15 | part-whole | stated | modified decoder with additional cross-attention | slide 65 |
| C17 | `uses` | C04 | functional | stated | cross-attention from decoder to final encoder output | slide 66 |
| C15 | `uses` | C18 | functional | stated | vocabulary probability output | J&M §7.5 |
| C19 | `uses` | C10 | functional | stated | parallel training without future access | J&M §7.7 |
| C19 | `uses` | C18 | functional | stated | score true next targets | J&M §7.7 |
| C20 | `assesses` | C15 | functional | stated | held-out next-token prediction | J&M §7.7.1 |
| C22 | `adjusts` | C21 | functional | stated | sampling distribution via logits | J&M §7.6.3 |
| C23 | `method_for` | C21 | functional | stated | truncated token-selection distributions | J&M §7.6.4 |
| C23 | `contrasts_with` | C22 | functional | inferred | truncate support versus rescale finite logits | J&M §7.6.3–7.6.4 |
| C24 | `property_of` | C06 | functional | stated | all-pairs dense attention | slides 69–70 |
| C25 | `method_for` | C21 | functional | stated | efficient autoregressive inference | J&M §7.8 |
| C25 | `uses` | C07 | functional | stated | new-query matrix attention with retained K/V | J&M Figure 7.23 |
| C26 | `uses` | C06 | functional | stated | local image lookup | Parmar §3 |
| C26 | `uses` | C10 | functional | stated | ungenerated pixels excluded | Parmar §3.3 |
| C27 | `uses` | C08 | functional | stated | learned relative offsets | Huang §3.3 |
| C27 | `uses` | C11 | functional | stated | multi-head musical sequence attention | Huang §3.2–3.3 |
| C28 | `method_for` | B09 | functional | stated | historical next-word prediction | J&M Historical Notes |
