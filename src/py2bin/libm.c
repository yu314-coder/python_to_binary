/* A libm written in the C py2bin compiles, so the compiler supplies the math
 * functions by compiling them rather than by linking anything. Every routine
 * reduces its argument to a small interval and evaluates a polynomial there;
 * the reductions use a split constant (Cody-Waite) so the reduced argument
 * keeps full precision. */

union __PY2BIN_BITS { double d; long long i; };

double __py2bin_scalbn(double v, long long k)
{
    /* 2^k built straight from the exponent field: no library, no loop. */
    union __PY2BIN_BITS u;
    if (k > 1023) { k = 1023; }
    if (k < -1022) { k = -1022; }
    u.i = (k + 1023) << 52;
    return v * u.d;
}

double exp(double x)
{
    if (x != x) { return x; }
    if (x > 709.782712893384) { return 1.0e308 * 10.0; }
    if (x < -745.1332191019411) { return 0.0; }
    double n = x * 1.4426950408889634;              /* x / ln 2 */
    long long k = (long long)(n < 0.0 ? n - 0.5 : n + 0.5);
    double kd = (double)k;
    /* ln 2 split so k * ln2 is exact to more than 53 bits. */
    double r = x - kd * 0.693147180369123816490 - kd * 1.90821492927058770002e-10;
    double p = 1.0 + r * (1.0 + r * (0.5 + r * (0.1666666666666666574
        + r * (0.04166666666666666435 + r * (0.008333333333333333218
        + r * (0.001388888888888888785 + r * (0.0001984126984126984125
        + r * (0.00002480158730158730495 + r * (0.000002755731922398589
        + r * (2.75573192239858883e-07 + r * (2.50521083854417202e-08
        + r * (2.08767569878681002e-09 + r * (1.60590438368216133e-10
        + r * 1.14707455977297245e-11)))))))))))));
    return __py2bin_scalbn(p, k);
}

double log(double x)
{
    if (x != x) { return x; }
    if (x < 0.0) { return 0.0 / 0.0; }
    if (x == 0.0) { return -1.0 / 0.0; }
    union __PY2BIN_BITS u;
    u.d = x;
    long long k = ((u.i >> 52) & 2047) - 1023;
    if (k == -1023) {           /* subnormal: scale into the normal range */
        u.d = x * 4503599627370496.0;
        k = (((u.i >> 52) & 2047) - 1023) - 52;
    }
    u.i = (u.i & 4503599627370495LL) | (1023LL << 52);   /* m in [1, 2) */
    double m = u.d;
    if (m > 1.4142135623730951) { m = m * 0.5; k = k + 1; }
    /* log m = 2 atanh t with t = (m-1)/(m+1); |t| <= 0.1716 */
    double t = (m - 1.0) / (m + 1.0);
    double s = t * t;
    double p = 1.0 + s * (0.3333333333333333 + s * (0.2 + s * (0.14285714285714285
        + s * (0.1111111111111111 + s * (0.09090909090909091 + s * (0.07692307692307693
        + s * (0.06666666666666667 + s * (0.058823529411764705 + s * (0.05263157894736842
        + s * 0.047619047619047616)))))))));
    return (double)k * 0.6931471805599453094 + 2.0 * t * p;
}

double __py2bin_sinpoly(double r)
{
    double s = r * r;
    return r * (
        1.0 + s * (-0.16666666666666666 + s * (0.008333333333333333 + s * (-0.0001984126984126984 + s * (2.7557319223985893e-06 + s * (-2.505210838544172e-08 + s * (1.6059043836821613e-10 + s * (-7.647163731819816e-13 + s * (2.8114572543455206e-15 + s * (-8.22063524662433e-18))))))))));
}

double __py2bin_cospoly(double r)
{
    double s = r * r;
    return (
        1.0 + s * (-0.5 + s * (0.041666666666666664 + s * (-0.001388888888888889 + s * (2.48015873015873e-05 + s * (-2.755731922398589e-07 + s * (2.08767569878681e-09 + s * (-1.1470745597729725e-11 + s * (4.779477332387385e-14 + s * (-1.5619206968586225e-16))))))))));
}

double __py2bin_trig(double x, long long want_cos)
{
    if (x != x) { return x; }
    if (x < -1.0e9 || x > 1.0e9) { return 0.0 / 0.0; }
    double n = x * 0.6366197723675814;              /* x / (pi/2) */
    long long k = (long long)(n < 0.0 ? n - 0.5 : n + 0.5);
    double kd = (double)k;
    /* pi/2 in three pieces, so the reduced argument stays accurate. */
    /* pi/2 split so that kd * (each part) is exact: the leading part has 33
     * significant bits and the rest are zero, which is what stops the
     * subtraction from cancelling away the low bits of r for large x. */
    /* pi/2 as a leading part with 33 significant bits (so kd * it is exact)
     * plus its tail. Subtracting a third piece here would double-count the
     * tail: the further stages of the classic reduction refine the REMAINDER,
     * they are not additional terms. */
    double r = x - kd * 1.57079632673412561417e+00;
    r = r - kd * 6.07710050650619224932e-11;
    long long q = (k + (want_cos ? 1 : 0)) & 3;
    if (q < 0) { q = q + 4; }
    if (q == 0) { return __py2bin_sinpoly(r); }
    if (q == 1) { return __py2bin_cospoly(r); }
    if (q == 2) { return -__py2bin_sinpoly(r); }
    return -__py2bin_cospoly(r);
}

double sin(double x) { return __py2bin_trig(x, 0); }
double cos(double x) { return __py2bin_trig(x, 1); }
double tan(double x) { return sin(x) / cos(x); }

double pow(double x, double y)
{
    if (y == 0.0) { return 1.0; }
    /* An integral exponent is exact by repeated squaring; going through
     * exp(y log x) would lose the last bit or two. */
    double ay = y < 0.0 ? -y : y;
    if (ay <= 1024.0 && ay == (double)(long long)ay && x == x) {
        long long e = (long long)ay;
        double base = x;
        double acc = 1.0;
        while (e > 0) {
            if (e & 1) { acc = acc * base; }
            base = base * base;
            e = e >> 1;
        }
        return y < 0.0 ? 1.0 / acc : acc;
    }
    if (x != x || y != y) { return 0.0 / 0.0; }
    if (x == 0.0) { return y < 0.0 ? 1.0 / 0.0 : 0.0; }
    if (x > 0.0) { return exp(y * log(x)); }
    /* A negative base is defined only for an integral exponent. */
    double t = y < 0.0 ? -y : y;
    if (t != (double)(long long)t) { return 0.0 / 0.0; }
    double magnitude = exp(y * log(-x));
    return ((long long)t & 1) ? -magnitude : magnitude;
}
