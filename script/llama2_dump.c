/* Reference activations for the Llama 2 forward pass (docs/plans/PLAN-llama2.md stage 1).
 *
 * llama2.c has no equivalent of llm.c's gpt2_124M_debug_state.bin, so this
 * builds one. run.c is included whole -- rmsnorm, softmax and matmul are used
 * as they are, so the arithmetic is the reference's, not a reimplementation.
 * Only forward()'s control flow is written out again, with each intermediate
 * recorded as it is produced.
 *
 * That duplication is the one risk here, and it is checked: the same binary
 * also greedily generates, and script/llama2_dump.rb requires the token
 * sequence to match what llama2.c's own test_all.py publishes for stories260K.
 *
 *   cc -O2 -o llama2_dump script/llama2_dump.c -lm -Ivendor/llama2.c
 *   ./llama2_dump data/stories260K.bin out.bin 8
 */

#define main llama2c_main_unused
#include "run.c"
#undef main

#define DUMP_MAGIC 20260916
#define DUMP_VERSION 1
#define DUMP_NAME_BYTES 24

/* One record per tensor: a fixed-width name, where it came from, and the
 * floats. Self-describing so the reader does not need a layout table. */
static void dump_tensor(FILE *out, const char *name, int pos, int layer,
                        float *data, int count) {
    char padded[DUMP_NAME_BYTES];
    memset(padded, 0, sizeof(padded));
    strncpy(padded, name, sizeof(padded) - 1);
    fwrite(padded, 1, sizeof(padded), out);
    fwrite(&pos, sizeof(int), 1, out);
    fwrite(&layer, sizeof(int), 1, out);
    fwrite(&count, sizeof(int), 1, out);
    fwrite(data, sizeof(float), count, out);
}

/* run.c:231-332 with the intermediates written out. Every line that computes
 * something is the original; the dump_tensor calls are the only additions. */
static float *forward_dump(Transformer *transformer, int token, int pos, FILE *out) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    float *content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));
    dump_tensor(out, "embed", pos, -1, x, dim);

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);
        dump_tensor(out, "rms_att", pos, (int)l, s->xb, dim);

        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);
        dump_tensor(out, "q_pre_rope", pos, (int)l, s->q, dim);
        dump_tensor(out, "k_pre_rope", pos, (int)l, s->k, kv_dim);
        dump_tensor(out, "v", pos, (int)l, s->v, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float *vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
        dump_tensor(out, "q", pos, (int)l, s->q, dim);
        dump_tensor(out, "k", pos, (int)l, s->k, kv_dim);

        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_size;
            float *att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        /* att is [n_heads, seq_len] but only the first pos+1 of each row is
         * live, so it is packed down before writing. */
        for (int hh = 0; hh < p->n_heads; hh++) {
            memmove(s->att + hh * (pos + 1), s->att + hh * p->seq_len,
                    (pos + 1) * sizeof(float));
        }
        dump_tensor(out, "att", pos, (int)l, s->att, p->n_heads * (pos + 1));
        dump_tensor(out, "attn_out", pos, (int)l, s->xb, dim);

        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        dump_tensor(out, "attproj", pos, (int)l, s->xb2, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }
        dump_tensor(out, "res_att", pos, (int)l, x, dim);

        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
        dump_tensor(out, "rms_ffn", pos, (int)l, s->xb, dim);

        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        dump_tensor(out, "w1h", pos, (int)l, s->hb, hidden_dim);
        dump_tensor(out, "w3h", pos, (int)l, s->hb2, hidden_dim);

        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        dump_tensor(out, "swiglu", pos, (int)l, s->hb, hidden_dim);

        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);
        dump_tensor(out, "ffn_out", pos, (int)l, s->xb, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
        dump_tensor(out, "res_ffn", pos, (int)l, x, dim);
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    dump_tensor(out, "rms_final", pos, -1, x, dim);

    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    dump_tensor(out, "logits", pos, -1, s->logits, p->vocab_size);
    return s->logits;
}

static int argmax_of(float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) { best = i; }
    }
    return best;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <checkpoint> <out.bin> <steps> [first_token]\n", argv[0]);
        return 1;
    }
    char *checkpoint = argv[1];
    char *out_path = argv[2];
    int steps = atoi(argv[3]);
    /* 1 is BOS, which is where test_all.py starts. */
    int first_token = argc > 4 ? atoi(argv[4]) : 1;

    Transformer transformer;
    build_transformer(&transformer, checkpoint);
    Config *p = &transformer.config;
    if (steps <= 0 || steps > p->seq_len) {
        fprintf(stderr, "steps must be in 1..%d\n", p->seq_len);
        return 1;
    }

    FILE *out = fopen(out_path, "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }

    int magic = DUMP_MAGIC, version = DUMP_VERSION;
    fwrite(&magic, sizeof(int), 1, out);
    fwrite(&version, sizeof(int), 1, out);
    fwrite(&p->dim, sizeof(int), 1, out);
    fwrite(&p->hidden_dim, sizeof(int), 1, out);
    fwrite(&p->n_layers, sizeof(int), 1, out);
    fwrite(&p->n_heads, sizeof(int), 1, out);
    fwrite(&p->n_kv_heads, sizeof(int), 1, out);
    fwrite(&p->vocab_size, sizeof(int), 1, out);
    fwrite(&p->seq_len, sizeof(int), 1, out);
    fwrite(&steps, sizeof(int), 1, out);

    /* The token at each position is written before its activations, so the
     * reader never has to guess what was fed in. */
    long tokens_at = ftell(out);
    int *tokens = calloc(steps, sizeof(int));
    fwrite(tokens, sizeof(int), steps, out);

    /* forward_dump is a copy of forward, so it is checked against the original
     * on a second transformer with its own state: same token, same position,
     * and the logits have to agree exactly. */
    Transformer control;
    build_transformer(&control, checkpoint);
    float worst = 0.0f;

    int token = first_token;
    for (int pos = 0; pos < steps; pos++) {
        tokens[pos] = token;
        float *logits = forward_dump(&transformer, token, pos, out);
        float *reference = forward(&control, token, pos);
        for (int i = 0; i < p->vocab_size; i++) {
            float d = fabsf(logits[i] - reference[i]);
            if (d > worst) { worst = d; }
        }
        token = argmax_of(logits, p->vocab_size);
    }
    free_transformer(&control);

    fseek(out, tokens_at, SEEK_SET);
    fwrite(tokens, sizeof(int), steps, out);
    fclose(out);

    fprintf(stderr, "wrote %s\n", out_path);
    printf("forward_dump vs forward: max|d| = %.9g%s\n", worst,
           worst == 0.0f ? " (exact)" : " NOT EXACT");
    printf("tokens:");
    for (int i = 0; i < steps; i++) { printf(" %d", tokens[i]); }
    printf("\nnext: %d\n", token);

    free(tokens);
    free_transformer(&transformer);
    return 0;
}
