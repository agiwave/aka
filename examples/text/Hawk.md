# Formula

```
y1  = a1 * s0 +  (1-a1) * x1
    = a1 * (s0 - x1) + x1
y2  = a2 * y1 + (1-a2) * x2
    = a2 * (y1-x2) + x2
    = a2 * ( a1 * (s0-x1) + x1 - x2) + x2
    = a1 * a2 * (s0-x1) + a2 * (x1-x2) + x2
yn  = a1 * a2 *...* a(n) * (s0-x1) + a2 * ... * a(n) * (x1-x2) + ... + xn
cuma = [a1, a1*a2, ..., a1*a2*...*an]
shiftx = [s0-x1, x1-x2, ...., x(n-1)-x(n)]
shifta = [1, cuma(0), cuma(1), ..., cuma(n-1)]
yn = cuma(n) / shifta(1) * shiftx(1) + .... + cuma(n) / shifta(n) * shiftx(n) + x(n)
```
