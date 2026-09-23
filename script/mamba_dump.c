/* Reference activations for the Mamba forward pass (docs/plans/PLAN-mamba.md stage 1).
 *
 * mamba.c ships no debug dump, and its CLI encodes the prompt with a 50277
 * entry tokenizer, so it cannot be pointed at a small checkpoint at all. This
 * takes token ids directly and writes every intermediate out.
 *
 * mamba.c is included whole -- rmsnorm, matmul, softplus and the rest are used
 * as they are, so the arithmetic is the reference's. Only forward_layer() and
 * forward()'s control flow are written out again, with each intermediate
 * recorded as it is produced.
 *
 * That duplication is the one risk here, and it is checked: a second model
 * with its own state runs the original forward() on the same tokens, and the
 * logits have to agree exactly.
 *
 *   cc -O2 -o mamba_dump script/mamba_dump.c -lm -Ivendor/mamba.c
 *   ./mamba_dump data/mamba_tiny.bin out.bin 4
 *   ./mamba_dump data/mamba-130m.bin out.bin 4 9038
 *   ./mamba_dump data/mamba_tiny.bin out.bin 0 3,17,42,8
 *   ./mamba_dump data/mamba-130m.bin - 256          # tokens only, no records
 *
 * The last form feeds the sequence as given instead of decoding greedily,
 * which a small random checkpoint needs: its argmax is a fixed point at
 * whatever token went in, so greedy decoding never varies the input.
 */

#define TESTING
#include "mamba.c"

#define DUMP_MAGIC 20260919
#define DUMP_VERSION 1
#define DUMP_NAME_BYTES 24

/* One record per tensor: a fixed-width name, where it came from, and the
 * floats. Self-describing so the reader does not need a layout table. */
static void dump_tensor(FILE *out, const char *name, int pos, int layer,
                        float *data, int count) {
    /* 256 steps of mamba-130m would be 2.3 GB of activations, and the token
     * sequence alone is what stage 2 needs. "-" asks for that. */
    if (out == NULL) { return; }
    char padded[DUMP_NAME_BYTES];
    memset(padded, 0, sizeof(padded));
    strncpy(padded, name, sizeof(padded) - 1);
    fwrite(padded, 1, sizeof(padded), out);
    fwrite(&pos, sizeof(int), 1, out);
    fwrite(&layer, sizeof(int), 1, out);
    fwrite(&count, sizeof(int), 1, out);
    fwrite(data, sizeof(float), count, out);
}

/* mamba.c:377 forward_layer with the intermediates written out. Every line
 * that computes something is the original; the dump_tensor calls are the only
 * additions. */
static void forward_layer_dump(Mamba *mamba, unsigned long long l,
                               float *hidden_state, int pos, FILE *out) {
    Config *p = &mamba->config;
    MambaWeights *w = &mamba->weights;
    RunState *s = &mamba->state;
    int dim = p->dim, d_inner = p->d_inner, d_conv = p->d_conv;
    int d_state = p->d_state, dt_rank = p->dt_rank;
    float *dA = s->dA;
    float *dB = s->dB;
    float *y = s->y;

    float *conv_state = s->conv_state + l * d_inner * d_conv;
    float *ssm_state = s->ssm_state + l * d_inner * d_state;

    float *in_proj = w->in_proj + l * 2 * d_inner * dim;
    float *conv1d_weight = w->conv1d_weight + l * d_inner * d_conv;
    float *conv1d_bias = w->conv1d_bias + l * d_inner;
    float *x_proj = w->x_proj + l * (dt_rank + 2 * d_state) * d_inner;
    float *dt_proj_weight = w->dt_proj_weight + l * d_inner * dt_rank;
    float *dt_proj_bias = w->dt_proj_bias + l * d_inner;
    float *A = w->A + l * d_inner * d_state;
    float *D = w->D + l * d_inner;
    float *out_proj = w->out_proj + l * dim * d_inner;

    matmul(s->xz, hidden_state, in_proj, 2 * d_inner, dim);
    dump_tensor(out, "xz", pos, (int)l, s->xz, 2 * d_inner);
    float *x = s->xz;
    float *z = s->xz + d_inner;

    shift_matrix_left(conv_state, d_inner, d_conv);
    update_last_column(conv_state, x, d_inner, d_conv);
    dump_tensor(out, "conv_state", pos, (int)l, conv_state, d_inner * d_conv);

    elementwise_multiply(s->temp, conv_state, conv1d_weight, d_inner * d_conv);
    sum_along_last_dim(x, s->temp, d_inner, d_conv);
    elementwise_add(x, x, conv1d_bias, d_inner);
    dump_tensor(out, "conv_out", pos, (int)l, x, d_inner);

    for (int i = 0; i < d_inner; i++) {
        x[i] = silu(x[i]);
    }
    dump_tensor(out, "x_silu", pos, (int)l, x, d_inner);

    matmul(s->x_db, x, x_proj, dt_rank + 2 * d_state, d_inner);
    dump_tensor(out, "x_db", pos, (int)l, s->x_db, dt_rank + 2 * d_state);
    float *dt = s->x_db;
    float *B = s->x_db + dt_rank;
    float *C = s->x_db + dt_rank + d_state;

    linear(s->dt, dt, dt_proj_weight, dt_proj_bias, d_inner, dt_rank);
    dt = s->dt;
    for (int i = 0; i < d_inner; i++) {
        dt[i] = softplus(dt[i]);
    }
    dump_tensor(out, "dt", pos, (int)l, dt, d_inner);

    broadcast_multiply(dA, dt, A, d_inner, d_state);
    for (int i = 0; i < d_inner * d_state; i++) {
        dA[i] = expf(dA[i]);
    }
    dump_tensor(out, "dA", pos, (int)l, dA, d_inner * d_state);

    outer_product(dB, dt, B, d_inner, d_state);
    dump_tensor(out, "dB", pos, (int)l, dB, d_inner * d_state);

    broadcast_multiply(s->temp, x, dB, d_inner, d_state);
    elementwise_multiply_and_add(ssm_state, ssm_state, dA, s->temp, d_inner * d_state);
    dump_tensor(out, "ssm_state", pos, (int)l, ssm_state, d_inner * d_state);

    rowwise_dot_product(y, ssm_state, C, d_inner, d_state);
    elementwise_multiply_and_add(y, D, x, y, d_inner);
    for (int i = 0; i < d_inner; i++) {
        y[i] = y[i] * silu(z[i]);
    }
    dump_tensor(out, "y", pos, (int)l, y, d_inner);

    matmul(hidden_state, y, out_proj, dim, d_inner);
    dump_tensor(out, "block_out", pos, (int)l, hidden_state, dim);
}

/* mamba.c:468 forward, likewise. */
static float *forward_dump(Mamba *mamba, int token, int pos, FILE *out) {
    Config *p = &mamba->config;
    MambaWeights *w = &mamba->weights;
    RunState *s = &mamba->state;
    int dim = p->dim;
    float *input = s->input;
    float *hidden_state = s->hidden_state;

    float *content_row = w->token_embedding_table + token * dim;
    memcpy(input, content_row, dim * sizeof(float));
    dump_tensor(out, "embed", pos, -1, input, dim);

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        rmsnorm(hidden_state, input, w->norm + l * dim, dim);
        dump_tensor(out, "rms", pos, (int)l, hidden_state, dim);

        forward_layer_dump(mamba, l, hidden_state, pos, out);

        for (int i = 0; i < dim; i++) {
            hidden_state[i] += input[i];
            input[i] = hidden_state[i];
        }
        dump_tensor(out, "residual", pos, (int)l, hidden_state, dim);
    }

    rmsnorm(hidden_state, hidden_state, w->final_norm, dim);
    dump_tensor(out, "rms_final", pos, -1, hidden_state, dim);

    matmul(s->logits, hidden_state, w->lm_head, p->rounded_vocab_size, p->dim);
    dump_tensor(out, "logits", pos, -1, s->logits, p->rounded_vocab_size);
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
        fprintf(stderr, "usage: %s <checkpoint> <out.bin> <steps> [first_token | id,id,...]\n",
                argv[0]);
        return 1;
    }
    char *checkpoint = argv[1];
    char *out_path = argv[2];
    int steps = atoi(argv[3]);
    /* 0 is <|endoftext|>, which is what mamba.c prepends. */
    int first_token = 0;
    int *given = NULL;
    int n_given = 0;
    if (argc > 4) {
        if (strchr(argv[4], ',')) {
            for (char *c = argv[4]; *c; c++) { if (*c == ',') { n_given++; } }
            n_given++;
            given = calloc(n_given, sizeof(int));
            char *copy = strdup(argv[4]);
            int i = 0;
            for (char *tok = strtok(copy, ","); tok; tok = strtok(NULL, ",")) {
                given[i++] = atoi(tok);
            }
            free(copy);
            steps = n_given;
        } else {
            first_token = atoi(argv[4]);
        }
    }

    Mamba mamba;
    load_model(&mamba, checkpoint);
    Config *p = &mamba.config;
    if (steps <= 0) { fprintf(stderr, "steps must be positive\n"); return 1; }
    for (int i = 0; i < n_given; i++) {
        if (given[i] < 0 || given[i] >= p->rounded_vocab_size) {
            fprintf(stderr, "token %d is outside 0..%d\n", given[i], p->rounded_vocab_size - 1);
            return 1;
        }
    }
    if (first_token < 0 || first_token >= p->rounded_vocab_size) {
        fprintf(stderr, "first_token must be in 0..%d\n", p->rounded_vocab_size - 1);
        return 1;
    }

    int tokens_only = strcmp(out_path, "-") == 0;
    FILE *out = NULL;
    long tokens_at = 0;
    if (!tokens_only) {
        out = fopen(out_path, "wb");
        if (!out) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }

        int magic = DUMP_MAGIC, version = DUMP_VERSION;
        fwrite(&magic, sizeof(int), 1, out);
        fwrite(&version, sizeof(int), 1, out);
        fwrite(&p->n_layers, sizeof(int), 1, out);
        fwrite(&p->vocab_size, sizeof(int), 1, out);
        fwrite(&p->dim, sizeof(int), 1, out);
        fwrite(&p->d_inner, sizeof(int), 1, out);
        fwrite(&p->dt_rank, sizeof(int), 1, out);
        fwrite(&p->d_state, sizeof(int), 1, out);
        fwrite(&p->d_conv, sizeof(int), 1, out);
        fwrite(&p->shared_classifier, sizeof(int), 1, out);
        fwrite(&steps, sizeof(int), 1, out);

        tokens_at = ftell(out);
    }
    int *tokens = calloc(steps, sizeof(int));
    if (out) { fwrite(tokens, sizeof(int), steps, out); }

    /* forward_dump is a copy of forward, so it is checked against the original
     * on a second model with its own state: same tokens in the same order, and
     * the logits have to agree exactly. Mamba carries its state forward, so
     * both have to be reset and then driven in step. */
    Mamba control;
    load_model(&control, checkpoint);
    reset_internal_state(&mamba);
    reset_internal_state(&control);
    float worst = 0.0f;

    int token = given ? given[0] : first_token;
    for (int pos = 0; pos < steps; pos++) {
        tokens[pos] = token;
        float *logits = forward_dump(&mamba, token, pos, out);
        float *reference = forward(&control, token);
        for (int i = 0; i < p->rounded_vocab_size; i++) {
            float d = fabsf(logits[i] - reference[i]);
            if (d > worst) { worst = d; }
        }
        token = given ? (pos + 1 < n_given ? given[pos + 1] : token)
                      : argmax_of(logits, p->vocab_size);
    }
    free_model(&control);
    free(given);

    if (out) {
        fseek(out, tokens_at, SEEK_SET);
        fwrite(tokens, sizeof(int), steps, out);
        fclose(out);
        fprintf(stderr, "wrote %s\n", out_path);
    }

    printf("forward_dump vs forward: max|d| = %.9g%s\n", worst,
           worst == 0.0f ? " (exact)" : " NOT EXACT");
    printf("tokens:");
    for (int i = 0; i < steps; i++) { printf(" %d", tokens[i]); }
    printf("\nnext: %d\n", token);

    free(tokens);
    free_model(&mamba);
    return 0;
}
