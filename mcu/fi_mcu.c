/* Freeze Index on a Cortex-M4F: bare-metal, no libc, single precision.
   4-s causal window (256 samples at 64 Hz), decision every 16 samples (0.25 s).
   Band powers by Goertzel on bins 2..32 (0.25 Hz resolution), Hann window, mean removal, 3 axes. */
#include <stdint.h>
#include "fi_tables.h"
#define W 256
#define HOP 16
#define K0 2
#define KS 12   /* bins 2..11 = 0.5-<3 Hz (locomotor), 12..32 = 3-8 Hz (freeze) */
#define K1 32

static float buf[3][W];
static uint32_t head, count;
float fi_out, lp_out, lloc_out;

static float ln_f(float x){                 /* natural log, |rel err| < 1e-7 for x > 0 */
  union { float f; uint32_t u; } v = { x };
  int e = (int)((v.u >> 23) & 0xFF) - 127;
  v.u = (v.u & 0x007FFFFF) | 0x3F800000;    /* mantissa in [1,2) */
  float m = v.f;
  if (m > 1.41421356f) { m *= 0.5f; e += 1; }
  float t = (m - 1.0f) / (m + 1.0f), t2 = t * t;
  float s = t * (2.0f + t2 * (0.6666667f + t2 * (0.4f + t2 * (0.2857143f + t2 * 0.2222222f))));
  return s + (float)e * 0.69314718f;
}

static float tmp[W];                        /* windowed, mean-removed copy of one axis */

static void band(float *frz, float *loc){
  float L = 0, F = 0;
  for (int a = 0; a < 3; a++){
    const float *b = buf[a];
    float m = 0; for (int i = 0; i < W; i++) m += b[i]; m *= (1.0f / W);
    uint32_t h = head;                      /* oldest sample first */
    for (int i = 0; i < W; i++){ tmp[i] = (b[h] - m) * HANN[i]; h = (h + 1) & (W - 1); }
    for (int k = K0; k <= K1; k++){
      float c = COSK[k], s1 = 0, s2 = 0;
      for (int i = 0; i < W; i++){ float s0 = tmp[i] + c * s1 - s2; s2 = s1; s1 = s0; }
      float p = s1 * s1 + s2 * s2 - c * s1 * s2;
      if (k < KS) L += p; else F += p;
    }
  }
  *frz = F; *loc = L;
}

void fi_init(float x, float y, float z){
  for (int i = 0; i < W; i++){ buf[0][i] = x; buf[1][i] = y; buf[2][i] = z; }
  head = 0; count = 0;
}

/* push one sample; returns 1 when a new decision (fi_out, lp_out) is ready */
int fi_push(float x, float y, float z){
  buf[0][head] = x; buf[1][head] = y; buf[2][head] = z;
  head = (head + 1) & (W - 1);              /* head now points to the oldest sample */
  int ready = (count % HOP) == 0;
  count++;
  if (ready){
    float F, L; band(&F, &L);
    fi_out = ln_f((F + 1e-8f) / (L + 1e-8f));
    lp_out = ln_f(F + L + 1e-8f);
    lloc_out = ln_f(L + 1e-8f);
  }
  return ready;
}

/* ---------- cue controller (D3, optional D4 arrest rule) ---------- */
typedef struct {
  float on, off, gate;          /* hysteresis thresholds on smoothed log FI, gate on log total power */
  int32_t k, hold;              /* smoothing length and minimum cue length, in decisions */
  int32_t use_arrest;           /* 1 = D4 */
  float drop, walk;             /* arrest: fall in locomotor log power, and minimum past level */
  int32_t arrest_hold;
} ctrl_params;

#define KMAX 16
#define PAST_A 40               /* arrest reference window: decisions i-40 .. i-17 */
#define PAST_B 16
static float fi_hist[KMAX];
#define RING (PAST_A + 1)
static float lloc_hist[RING];
static int32_t nd, hyst, hold_left, arr_left;

void ctrl_init(void){ nd = 0; hyst = 0; hold_left = 0; arr_left = 0; }

static float median_past(void){  /* median of lloc for decisions max(0,i-40) .. max(0,i-16)-1 */
  int32_t lo = nd - PAST_A; if (lo < 0) lo = 0;
  int32_t hi = nd - PAST_B; if (hi < 0) hi = 0;
  int32_t m = hi - lo;
  if (m <= 0) return lloc_hist[nd % RING];
  float v[PAST_A - PAST_B];
  for (int32_t j = 0; j < m; j++){            /* insertion sort, at most 24 values */
    float t = lloc_hist[(lo + j) % RING]; int32_t q = j;
    while (q > 0 && v[q-1] > t){ v[q] = v[q-1]; q--; }
    v[q] = t;
  }
  return (m & 1) ? v[m/2] : 0.5f * (v[m/2 - 1] + v[m/2]);
}

/* one decision: takes fi_out, lp_out, lloc_out of the current hop, returns cue on/off */
int ctrl_step(const ctrl_params *p){
  fi_hist[nd % KMAX] = fi_out;
  lloc_hist[nd % RING] = lloc_out;
  int32_t n = nd + 1 < p->k ? nd + 1 : p->k;
  float x = 0; for (int32_t j = 0; j < n; j++) x += fi_hist[(nd - j) % KMAX]; x /= (float)n;
  int32_t gated = !(lp_out > p->gate);                    /* gate closed -> x = -inf */
  int32_t prev = hyst;
  if (!gated && x > p->on) hyst = 1;
  else if (gated || x < p->off) hyst = 0;
  if (hyst && !prev) hold_left = p->hold;                 /* minimum cue length from each start */
  int32_t cue = hyst || hold_left > 0;
  if (hold_left > 0) hold_left--;
  if (p->use_arrest){
    float past = median_past();
    float arrest = lloc_out - past;
    float past_level = lp_out - arrest;
    int32_t raw = (-arrest > p->drop) && (past_level > p->walk);
    static int32_t raw_prev;
    if (nd == 0) raw_prev = 0;
    if (raw && !raw_prev) arr_left = p->arrest_hold;
    int32_t a = raw || arr_left > 0;
    if (arr_left > 0) arr_left--;
    raw_prev = raw;
    cue = cue || a;
  }
  nd++;
  return cue;
}

/* test entry used by the emulator: x = n interleaved xyz samples; out = fi, logP per decision; cue = 1 byte per decision */
int run(const float *x, int n, float *out, uint8_t *cue, const ctrl_params *p){
  fi_init(x[0], x[1], x[2]);
  ctrl_init();
  int h = 0;
  for (int j = 0; j < n; j++)
    if (fi_push(x[3*j], x[3*j+1], x[3*j+2])){
      out[2*h] = fi_out; out[2*h+1] = lp_out;
      cue[h] = (uint8_t)ctrl_step(p);
      h++;
    }
  return h;
}
